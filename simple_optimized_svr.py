import pandas as pd
import numpy as np
import os
import pickle
import argparse
import time
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from tqdm import tqdm
from utils import parse_mutation, read_fasta, BLOSUM62, AA_PROPERTIES, write_top_mutants

class SimpleOptimizedSVR:
    def __init__(self, data_dir, models_dir, output_dir, window_size=21, select_features=100, random_state=42):
        """Initialize the optimized SVR model without ESM embeddings."""
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.output_dir = output_dir
        self.window_size = window_size
        self.select_features = select_features
        self.random_state = random_state
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs("Leaderboard", exist_ok=True)
        
        # Initialize data containers
        self.protein_sequence = None
        self.train_df = None
        self.val_df = None
        self.test_df = None
        
        # Initialize feature selectors and scalers
        self.feature_scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.feature_selector = SelectKBest(f_regression, k=select_features)
        
        # Initialize SVR model with default parameters (to be optimized)
        self.model = SVR()
    
    def load_data(self):
        """Load and preprocess data with very aggressive validation incorporation."""
        print("\nStep 1: Loading and preprocessing data...\n")
        
        # Load protein sequence
        sequence_file = os.path.join(self.data_dir, "sequence.fasta")
        self.protein_sequence = read_fasta(sequence_file)
        print(f"Loaded protein sequence of length {len(self.protein_sequence)}")
        
        # Load training data
        train_file = os.path.join(self.data_dir, "train.csv")
        all_train_df = pd.read_csv(train_file)
        print(f"Loaded {len(all_train_df)} training mutations")
        
        # Set aside a small validation set (only 10% validation)
        np.random.seed(self.random_state)
        val_indices = np.random.choice(all_train_df.index, size=int(len(all_train_df) * 0.1), replace=False)
        self.val_df = all_train_df.iloc[val_indices].reset_index(drop=True)
        self.train_df = all_train_df.drop(val_indices).reset_index(drop=True)
        
        print(f"Training set: {len(self.train_df)} samples")
        print(f"Validation set: {len(self.val_df)} samples")
        
        # Load test data
        test_file = os.path.join(self.data_dir, "test.csv")
        self.test_df = pd.read_csv(test_file)
        print(f"Test set: {len(self.test_df)} samples")
        
        return self.train_df, self.val_df, self.test_df, self.protein_sequence
    
    def extract_mutation_features(self, mutation_str):
        """Extract enhanced features for a single mutation."""
        # Parse mutation
        original_aa, position, new_aa = parse_mutation(mutation_str)
        
        # Check if position is valid
        if position < 0 or position >= len(self.protein_sequence):
            # For invalid positions, return a dictionary of zeros
            return {f"feature_{i}": 0 for i in range(100)}  # Arbitrary large number
        
        # Get BLOSUM62 score
        blosum62_score = BLOSUM62.get((original_aa, new_aa), 0)
        
        # Get amino acid property changes
        hydrophobic_aas = {'A', 'F', 'G', 'I', 'L', 'M', 'P', 'V', 'W'}
        polar_aas = {'C', 'N', 'Q', 'S', 'T', 'Y'}
        charged_aas = {'D', 'E', 'H', 'K', 'R'}
        aromatic_aas = {'F', 'W', 'Y', 'H'}
        aliphatic_aas = {'A', 'I', 'L', 'V'}
        
        # Secondary structure propensity groups
        helix_aas = {'A', 'E', 'L', 'M'}
        sheet_aas = {'V', 'I', 'Y', 'F', 'W', 'C'}
        turn_aas = {'G', 'P', 'S', 'N', 'D'}
        
        # Extract window around mutation
        half_window = self.window_size // 2
        window_start = max(0, position - half_window)
        window_end = min(len(self.protein_sequence), position + half_window + 1)
        window_seq = self.protein_sequence[window_start:window_end]
        
        # Get amino acid counts in window
        aa_counts = {}
        for aa in 'ACDEFGHIKLMNPQRSTVWY':
            aa_counts[aa] = window_seq.count(aa)
        
        # Calculate a conservation score (proportion of similar amino acids in window)
        conservation_score = sum(1 for aa in window_seq if BLOSUM62.get((original_aa, aa), 0) > 0) / len(window_seq)
        
        # Compute position-based features
        normalized_position = position / len(self.protein_sequence)
        n_terminal_region = 1 if normalized_position < 0.25 else 0
        c_terminal_region = 1 if normalized_position > 0.75 else 0
        
        # Get properties
        wild_props = AA_PROPERTIES.get(original_aa, {})
        mutant_props = AA_PROPERTIES.get(new_aa, {})
        
        # Calculate property changes
        hydrophobicity_change = mutant_props.get('hydrophobicity', 0) - wild_props.get('hydrophobicity', 0)
        volume_change = mutant_props.get('volume', 0) - wild_props.get('volume', 0)
        polarity_change = mutant_props.get('polarity', 0) - wild_props.get('polarity', 0)
        charge_change = mutant_props.get('charge', 0) - wild_props.get('charge', 0)
        h_bond_donors_change = mutant_props.get('h_bond_donors', 0) - wild_props.get('h_bond_donors', 0)
        h_bond_acceptors_change = mutant_props.get('h_bond_acceptors', 0) - wild_props.get('h_bond_acceptors', 0)
        helix_propensity_change = mutant_props.get('helix_propensity', 0) - wild_props.get('helix_propensity', 0)
        sheet_propensity_change = mutant_props.get('sheet_propensity', 0) - wild_props.get('sheet_propensity', 0)
        turn_propensity_change = mutant_props.get('turn_propensity', 0) - wild_props.get('turn_propensity', 0)
        
        # Create feature dictionary - ensure all keys are strings
        features = {
            "blosum62": blosum62_score,
            "hydrophobicity_change": hydrophobicity_change,
            "volume_change": volume_change,
            "polarity_change": polarity_change,
            "charge_change": charge_change,
            "h_bond_donors_change": h_bond_donors_change,
            "h_bond_acceptors_change": h_bond_acceptors_change,
            "helix_propensity_change": helix_propensity_change,
            "sheet_propensity_change": sheet_propensity_change,
            "turn_propensity_change": turn_propensity_change,
            "position": position,
            "position_normalized": normalized_position,
            "n_terminal_region": n_terminal_region,
            "c_terminal_region": c_terminal_region,
            "wild_type_is_hydrophobic": 1 if original_aa in hydrophobic_aas else 0,
            "wild_type_is_polar": 1 if original_aa in polar_aas else 0,
            "wild_type_is_charged": 1 if original_aa in charged_aas else 0,
            "wild_type_is_aromatic": 1 if original_aa in aromatic_aas else 0,
            "wild_type_is_aliphatic": 1 if original_aa in aliphatic_aas else 0,
            "mutant_is_hydrophobic": 1 if new_aa in hydrophobic_aas else 0,
            "mutant_is_polar": 1 if new_aa in polar_aas else 0,
            "mutant_is_charged": 1 if new_aa in charged_aas else 0,
            "mutant_is_aromatic": 1 if new_aa in aromatic_aas else 0,
            "mutant_is_aliphatic": 1 if new_aa in aliphatic_aas else 0,
            "conservation_score": conservation_score
        }
        
        # Add one-hot encoding for original and new amino acid
        for aa in 'ACDEFGHIKLMNPQRSTVWY':
            features[f"original_aa_{aa}"] = 1 if original_aa == aa else 0
            features[f"new_aa_{aa}"] = 1 if new_aa == aa else 0
        
        # Add interactions between features
        features["hydrophobicity_volume_interaction"] = hydrophobicity_change * volume_change
        features["position_conservation_interaction"] = normalized_position * conservation_score
        
        # Add squared terms for possible non-linear relationships
        features["position_squared"] = normalized_position ** 2
        features["hydrophobicity_change_squared"] = hydrophobicity_change ** 2
        features["volume_change_squared"] = volume_change ** 2
        
        return features
    
    def create_features(self):
        """Create enhanced features for training, validation, and test sets."""
        print("\nStep 2: Creating enhanced features...\n")
        
        # Define a common feature template to ensure consistent features across datasets
        feature_template = self.extract_mutation_features("A1G")  # Use a dummy mutation to create a template
        common_features = list(feature_template.keys())
        
        # Extract features for training data
        print("Extracting features for training data...")
        train_features_list = []
        for mutation in tqdm(self.train_df['mutant'], desc="Training mutations"):
            features = self.extract_mutation_features(mutation)
            # Ensure all features in the template are present
            for feature in common_features:
                if feature not in features:
                    features[feature] = 0
            train_features_list.append({k: features.get(k, 0) for k in common_features})
        
        self.train_features = pd.DataFrame(train_features_list)
        
        # Extract features for validation data
        print("Extracting features for validation data...")
        val_features_list = []
        for mutation in tqdm(self.val_df['mutant'], desc="Validation mutations"):
            features = self.extract_mutation_features(mutation)
            # Ensure all features in the template are present
            for feature in common_features:
                if feature not in features:
                    features[feature] = 0
            val_features_list.append({k: features.get(k, 0) for k in common_features})
        
        self.val_features = pd.DataFrame(val_features_list)
        
        # Ensure validation features have the same columns as train features
        missing_cols = set(self.train_features.columns) - set(self.val_features.columns)
        for col in missing_cols:
            self.val_features[col] = 0
        self.val_features = self.val_features[self.train_features.columns]
        
        # Extract features for test data
        print("Extracting features for test data...")
        test_features_list = []
        for mutation in tqdm(self.test_df['mutant'], desc="Test mutations"):
            features = self.extract_mutation_features(mutation)
            # Ensure all features in the template are present
            for feature in common_features:
                if feature not in features:
                    features[feature] = 0
            test_features_list.append({k: features.get(k, 0) for k in common_features})
        
        self.test_features = pd.DataFrame(test_features_list)
        
        # Ensure test features have the same columns as train features
        missing_cols = set(self.train_features.columns) - set(self.test_features.columns)
        for col in missing_cols:
            self.test_features[col] = 0
        self.test_features = self.test_features[self.train_features.columns]
        
        print(f"Feature dimensions - Train: {self.train_features.shape}, Val: {self.val_features.shape}, Test: {self.test_features.shape}")
        
        return self.train_features, self.val_features, self.test_features
    
    def prepare_features(self):
        """Prepare features for training by handling missing values, selecting top features, and scaling."""
        print("\nStep 3: Preparing features for training...\n")
        
        # Convert to numpy arrays
        X_train = self.train_features.values
        X_val = self.val_features.values
        X_test = self.test_features.values
        
        # Handle missing values
        X_train = self.imputer.fit_transform(X_train)
        X_val = self.imputer.transform(X_val)
        X_test = self.imputer.transform(X_test)
        
        # Get target values
        y_train = self.train_df['DMS_score'].values
        y_val = self.val_df['DMS_score'].values
        
        # Select top features if needed
        if self.select_features < X_train.shape[1]:
            print(f"Selecting top {self.select_features} features out of {X_train.shape[1]} using mutual information")
            feature_selector = SelectKBest(mutual_info_regression, k=self.select_features)
            X_train_selected = feature_selector.fit_transform(X_train, y_train)
            selected_indices = feature_selector.get_support(indices=True)
            
            # Print top 10 feature names
            feature_names = list(self.train_features.columns)
            selected_names = [feature_names[i] for i in selected_indices]
            print("Top 10 selected features:")
            for i, name in enumerate(selected_names[:10]):
                print(f"{i+1}. {name}")
            
            # Apply feature selection to validation and test data
            X_val_selected = feature_selector.transform(X_val)
            X_test_selected = feature_selector.transform(X_test)
        else:
            print(f"Using all {X_train.shape[1]} features")
            X_train_selected = X_train
            X_val_selected = X_val
            X_test_selected = X_test
        
        # Scale features
        X_train_scaled = self.feature_scaler.fit_transform(X_train_selected)
        X_val_scaled = self.feature_scaler.transform(X_val_selected)
        X_test_scaled = self.feature_scaler.transform(X_test_selected)
        
        print(f"Final feature dimensions - Train: {X_train_scaled.shape}, Val: {X_val_scaled.shape}, Test: {X_test_scaled.shape}")
        
        return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled
    
    def optimize_model(self, X_train, y_train, X_val, y_val):
        """Optimize SVR hyperparameters."""
        print("\nStep 4: Optimizing SVR hyperparameters...\n")
        
        # Define parameter space for SVR
        param_distributions = {
            'C': np.logspace(-3, 3, 7),  # [0.001, 0.01, 0.1, 1, 10, 100, 1000]
            'gamma': np.logspace(-5, 1, 7),  # [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]
            'epsilon': np.logspace(-3, 0, 4),  # [0.001, 0.01, 0.1, 1]
            'kernel': ['rbf', 'linear', 'poly']
        }
        
        # Define SVR model
        svr = SVR()
        
        # Define scoring function (Spearman correlation)
        def spearman_scorer(estimator, X, y):
            y_pred = estimator.predict(X)
            return spearmanr(y, y_pred)[0]
        
        # Use RandomizedSearchCV to find best parameters
        random_search = RandomizedSearchCV(
            svr,
            param_distributions=param_distributions,
            n_iter=20,  # Number of parameter combinations to try
            scoring=spearman_scorer,
            cv=5,  # 5-fold cross-validation
            verbose=1,
            random_state=self.random_state,
            n_jobs=-1  # Use all cores
        )
        
        # Fit random search to find best parameters
        random_search.fit(X_train, y_train)
        
        # Print best parameters
        print("Best parameters found:")
        for param, value in random_search.best_params_.items():
            print(f"  {param}: {value}")
        
        # Get best estimator
        best_svr = random_search.best_estimator_
        
        # Evaluate on validation data
        y_val_pred = best_svr.predict(X_val)
        val_corr, _ = spearmanr(y_val, y_val_pred)
        print(f"Validation Spearman correlation: {val_corr:.4f}")
        
        # Save the validation predictions for analysis
        val_preds_df = pd.DataFrame({
            'mutant': self.val_df['mutant'],
            'DMS_score': y_val,
            'DMS_score_predicted': y_val_pred
        })
        val_preds_file = os.path.join(self.output_dir, "validation_predictions.csv")
        val_preds_df.to_csv(val_preds_file, index=False)
        print(f"Saved validation predictions to {val_preds_file}")
        
        # Save best model
        self.model = best_svr
        model_file = os.path.join(self.models_dir, "simple_optimized_svr_model.pkl")
        with open(model_file, 'wb') as f:
            pickle.dump(best_svr, f)
        print(f"Saved optimized SVR model to {model_file}")
        
        return best_svr, val_corr
    
    def retrain_on_all_data(self, X_train, y_train, X_val, y_val):
        """Retrain the model on all data (train + validation)."""
        print("\nStep 5: Retraining final model on all available data...\n")
        
        # Combine training and validation data
        X_combined = np.vstack([X_train, X_val])
        y_combined = np.concatenate([y_train, y_val])
        
        # Create a new SVR with the same hyperparameters as the best model
        final_svr = SVR(**{key: value for key, value in self.model.get_params().items() 
                          if key in ['C', 'gamma', 'epsilon', 'kernel', 'degree', 'coef0']})
        
        # Train on all data
        print("Training final model on combined data...")
        final_svr.fit(X_combined, y_combined)
        
        # Save final model
        self.model = final_svr
        model_file = os.path.join(self.models_dir, "simple_final_svr_model.pkl")
        with open(model_file, 'wb') as f:
            pickle.dump(final_svr, f)
        print(f"Saved final SVR model to {model_file}")
        
        return final_svr
    
    def make_predictions(self, X_test):
        """Make predictions on the test set."""
        print("\nStep 6: Making final predictions...\n")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'mutant': self.test_df['mutant'],
            'DMS_score_predicted': y_pred
        })
        
        return results_df
    
    def save_results(self, results_df, top_n=10):
        """Save predictions and top mutants."""
        print("\nStep 7: Saving predictions and LeaderBoard files...\n")
        
        # Save all predictions
        predictions_file = os.path.join(self.output_dir, "predictions.csv")
        results_df.to_csv(predictions_file, index=False)
        print(f"Saved predictions to {predictions_file}")
        
        # Also save to Leaderboard directory
        leaderboard_file = "Leaderboard/predictions.csv"
        results_df.to_csv(leaderboard_file, index=False)
        print(f"Saved predictions to {leaderboard_file}")
        
        # Save top N mutants
        top_mutants_file = os.path.join(self.output_dir, "top10.txt")
        top_mutants = write_top_mutants(results_df, top_mutants_file, n=top_n)
        
        # Also save to Leaderboard directory
        leaderboard_top_file = "Leaderboard/top10.txt"
        write_top_mutants(results_df, leaderboard_top_file, n=top_n)
        
        print("\nTop predicted beneficial mutations:")
        print(top_mutants[['mutant', 'DMS_score_predicted']].to_string(index=False))
        
        return predictions_file, top_mutants_file
    
    def run_pipeline(self, top_n=10):
        """Run the complete pipeline."""
        # Record start time
        start_time = time.time()
        
        print("="*80)
        print("Simple Optimized SVR Model for Protein Mutation Effect Prediction")
        print("="*80)
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Create enhanced features
        self.create_features()
        
        # Step 3: Prepare features
        X_train, y_train, X_val, y_val, X_test = self.prepare_features()
        
        # Step 4: Optimize model
        self.optimize_model(X_train, y_train, X_val, y_val)
        
        # Step 5: Retrain on all data
        self.retrain_on_all_data(X_train, y_train, X_val, y_val)
        
        # Step 6: Make predictions
        results_df = self.make_predictions(X_test)
        
        # Step 7: Save results
        predictions_file, top_mutants_file = self.save_results(results_df, top_n=top_n)
        
        # Print runtime
        runtime_minutes = (time.time() - start_time) / 60
        print(f"\nTotal runtime: {runtime_minutes:.2f} minutes")
        
        return predictions_file, top_mutants_file

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Simple Optimized SVR Model for Protein Engineering")
    parser.add_argument("--data_dir", type=str, default="Hackathon_data", help="Directory containing data files")
    parser.add_argument("--models_dir", type=str, default="simple_optimized_svr_models", help="Directory to save models")
    parser.add_argument("--output_dir", type=str, default="simple_optimized_svr_results", help="Directory to save results")
    parser.add_argument("--window_size", type=int, default=21, help="Window size for feature extraction")
    parser.add_argument("--select_features", type=int, default=100, help="Number of top features to select")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument("--top_n", type=int, default=10, help="Number of top mutations to output")
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Initialize model
    model = SimpleOptimizedSVR(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        output_dir=args.output_dir,
        window_size=args.window_size,
        select_features=args.select_features,
        random_state=args.random_state
    )
    
    # Run pipeline
    model.run_pipeline(top_n=args.top_n)

if __name__ == "__main__":
    main() 