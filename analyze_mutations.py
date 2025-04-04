import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from utils import parse_mutation, AA_PROPERTIES, read_fasta, BLOSUM62
from collections import Counter

def analyze_top_mutations(predictions_file, sequence_file, top_n=50, output_dir="mutation_analysis"):
    """Analyze the top predicted mutations"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load predictions
    predictions = pd.read_csv(predictions_file)
    
    # Sort by predicted DMS score in descending order
    sorted_predictions = predictions.sort_values("DMS_score_predicted", ascending=False)
    
    # Get top N mutations
    top_mutations = sorted_predictions.head(top_n)
    
    # Load protein sequence
    protein_sequence = read_fasta(sequence_file)
    print(f"Loaded protein sequence of length {len(protein_sequence)}")
    
    # Parse mutations
    original_aas = []
    positions = []
    new_aas = []
    blosum_scores = []
    hydrophobicity_changes = []
    volume_changes = []
    charge_changes = []
    
    # Compute statistics for each mutation
    for mutation in top_mutations["mutant"]:
        original_aa, position, new_aa = parse_mutation(mutation)
        
        original_aas.append(original_aa)
        positions.append(position)
        new_aas.append(new_aa)
        
        # BLOSUM62 score
        blosum_scores.append(BLOSUM62.get((original_aa, new_aa), 0))
        
        # Property changes
        if original_aa in AA_PROPERTIES and new_aa in AA_PROPERTIES:
            hydrophobicity_changes.append(
                AA_PROPERTIES[new_aa]["hydrophobicity"] - AA_PROPERTIES[original_aa]["hydrophobicity"]
            )
            volume_changes.append(
                AA_PROPERTIES[new_aa]["volume"] - AA_PROPERTIES[original_aa]["volume"]
            )
            charge_changes.append(
                AA_PROPERTIES[new_aa]["charge"] - AA_PROPERTIES[original_aa]["charge"]
            )
        else:
            hydrophobicity_changes.append(0)
            volume_changes.append(0)
            charge_changes.append(0)
    
    # Add parsed information to the DataFrame
    top_mutations["original_aa"] = original_aas
    top_mutations["position"] = positions
    top_mutations["new_aa"] = new_aas
    top_mutations["blosum_score"] = blosum_scores
    top_mutations["hydrophobicity_change"] = hydrophobicity_changes
    top_mutations["volume_change"] = volume_changes
    top_mutations["charge_change"] = charge_changes
    
    # Save the enhanced DataFrame
    top_mutations_file = os.path.join(output_dir, "top_mutations_analyzed.csv")
    top_mutations.to_csv(top_mutations_file, index=False)
    print(f"Saved analyzed top mutations to {top_mutations_file}")
    
    # Analysis 1: Amino acid substitution patterns
    print("\nAnalyzing amino acid substitution patterns...")
    
    # Count original amino acids
    original_aa_counts = pd.Series(original_aas).value_counts()
    print("\nOriginal amino acid distribution:")
    print(original_aa_counts)
    
    # Count new amino acids
    new_aa_counts = pd.Series(new_aas).value_counts()
    print("\nNew amino acid distribution:")
    print(new_aa_counts)
    
    # Count specific substitutions
    substitutions = [f"{orig_aa}→{new_aa}" for orig_aa, new_aa in zip(original_aas, new_aas)]
    substitution_counts = pd.Series(substitutions).value_counts().head(10)
    print("\nTop substitution patterns:")
    print(substitution_counts)
    
    # Visualization: Create a substitution matrix heatmap
    substitution_matrix = pd.DataFrame(0, 
                                     index=sorted(set(original_aas)), 
                                     columns=sorted(set(new_aas)))
    
    for orig_aa, new_aa in zip(original_aas, new_aas):
        substitution_matrix.loc[orig_aa, new_aa] += 1
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(substitution_matrix, annot=True, cmap="YlGnBu", fmt="d")
    plt.title("Amino Acid Substitution Matrix for Top Mutations")
    plt.xlabel("New Amino Acid")
    plt.ylabel("Original Amino Acid")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "substitution_matrix.png"), dpi=300)
    
    # Analysis 2: Position distribution
    print("\nAnalyzing mutation positions...")
    
    plt.figure(figsize=(12, 6))
    plt.hist(positions, bins=20, alpha=0.7, color='blue')
    plt.title("Distribution of Mutation Positions")
    plt.xlabel("Position in Protein Sequence")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "position_distribution.png"), dpi=300)
    
    # Analysis 3: Property changes
    print("\nAnalyzing property changes...")
    
    property_changes = pd.DataFrame({
        "BLOSUM62 Score": blosum_scores,
        "Hydrophobicity Change": hydrophobicity_changes,
        "Volume Change": volume_changes,
        "Charge Change": charge_changes
    })
    
    print("\nAverage property changes:")
    print(property_changes.mean())
    
    plt.figure(figsize=(15, 10))
    for i, property_name in enumerate(property_changes.columns):
        plt.subplot(2, 2, i+1)
        plt.hist(property_changes[property_name], bins=10, alpha=0.7)
        plt.title(f"Distribution of {property_name}")
        plt.xlabel(property_name)
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "property_changes.png"), dpi=300)
    
    # Analysis 4: Relationship between property changes and predicted scores
    print("\nAnalyzing relationship between property changes and predicted scores...")
    
    combined_data = pd.concat([top_mutations["DMS_score_predicted"], property_changes], axis=1)
    
    plt.figure(figsize=(15, 10))
    for i, property_name in enumerate(property_changes.columns):
        plt.subplot(2, 2, i+1)
        plt.scatter(combined_data[property_name], combined_data["DMS_score_predicted"], alpha=0.7)
        plt.title(f"{property_name} vs Predicted Score")
        plt.xlabel(property_name)
        plt.ylabel("Predicted DMS Score")
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "property_vs_score.png"), dpi=300)
    
    # Summary of findings
    print("\nSummary of findings:")
    print(f"1. Most common original amino acids: {', '.join(original_aa_counts.index[:3])}")
    print(f"2. Most common new amino acids: {', '.join(new_aa_counts.index[:3])}")
    print(f"3. Most common substitution: {substitution_counts.index[0]} (occurs {substitution_counts.iloc[0]} times)")
    
    avg_score = top_mutations["DMS_score_predicted"].mean()
    print(f"4. Average predicted DMS score: {avg_score:.4f}")
    
    return top_mutations

def load_predictions(file_path):
    """Load prediction results from the CSV file."""
    return pd.read_csv(file_path)

def load_top_mutations(file_path):
    """Load top mutations from the text file."""
    mutations = []
    scores = []
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                mutations.append(parts[0])
                scores.append(float(parts[1]))
    
    return pd.DataFrame({'mutant': mutations, 'DMS_score_predicted': scores})

def parse_mutations_df(mutations_df):
    """Parse mutations into original AA, position, and new AA."""
    results = []
    
    for _, row in mutations_df.iterrows():
        mutation = row['mutant']
        score = row['DMS_score_predicted']
        
        original_aa, position, new_aa = parse_mutation(mutation)
        
        results.append({
            'mutant': mutation,
            'original_aa': original_aa,
            'position': position,
            'new_aa': new_aa,
            'score': score
        })
    
    return pd.DataFrame(results)

def analyze_positions(parsed_df, top_n=20):
    """Analyze the distribution of mutation positions."""
    position_counts = Counter(parsed_df['position'])
    top_positions = pd.DataFrame({
        'position': [pos for pos, _ in position_counts.most_common(top_n)],
        'count': [count for _, count in position_counts.most_common(top_n)]
    })
    
    return top_positions

def analyze_amino_acid_changes(parsed_df):
    """Analyze amino acid substitution patterns."""
    # Original amino acid distribution
    orig_aa_counts = Counter(parsed_df['original_aa'])
    orig_aa_df = pd.DataFrame({
        'amino_acid': list(orig_aa_counts.keys()),
        'count': list(orig_aa_counts.values())
    }).sort_values('count', ascending=False)
    
    # New amino acid distribution
    new_aa_counts = Counter(parsed_df['new_aa'])
    new_aa_df = pd.DataFrame({
        'amino_acid': list(new_aa_counts.keys()),
        'count': list(new_aa_counts.values())
    }).sort_values('count', ascending=False)
    
    # Amino acid substitution patterns
    substitutions = Counter(parsed_df.apply(lambda x: f"{x['original_aa']}→{x['new_aa']}", axis=1))
    subst_df = pd.DataFrame({
        'substitution': list(substitutions.keys()),
        'count': list(substitutions.values())
    }).sort_values('count', ascending=False).head(10)
    
    return orig_aa_df, new_aa_df, subst_df

def analyze_property_changes(parsed_df):
    """Analyze changes in amino acid properties."""
    properties = ['hydrophobicity', 'volume', 'polarity', 'charge', 'helix_propensity', 'sheet_propensity', 'turn_propensity']
    property_changes = []
    
    for _, row in parsed_df.iterrows():
        orig_aa = row['original_aa']
        new_aa = row['new_aa']
        
        if orig_aa in AA_PROPERTIES and new_aa in AA_PROPERTIES:
            changes = {}
            for prop in properties:
                orig_val = AA_PROPERTIES[orig_aa].get(prop, 0)
                new_val = AA_PROPERTIES[new_aa].get(prop, 0)
                changes[prop] = new_val - orig_val
            
            changes['mutant'] = row['mutant']
            changes['score'] = row['score']
            property_changes.append(changes)
    
    return pd.DataFrame(property_changes)

def plot_top_positions(top_positions, output_dir):
    """Plot the distribution of top mutation positions."""
    plt.figure(figsize=(12, 6))
    sns.barplot(x='position', y='count', data=top_positions)
    plt.title('Distribution of Top Mutation Positions')
    plt.xlabel('Position in Protein')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_positions.png'), dpi=300)
    plt.close()

def plot_aa_distributions(orig_aa_df, new_aa_df, output_dir):
    """Plot the distribution of original and new amino acids."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original amino acids
    sns.barplot(x='amino_acid', y='count', data=orig_aa_df, ax=ax1)
    ax1.set_title('Distribution of Original Amino Acids')
    ax1.set_xlabel('Amino Acid')
    ax1.set_ylabel('Count')
    
    # New amino acids
    sns.barplot(x='amino_acid', y='count', data=new_aa_df, ax=ax2)
    ax2.set_title('Distribution of New Amino Acids')
    ax2.set_xlabel('Amino Acid')
    ax2.set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'aa_distributions.png'), dpi=300)
    plt.close()

def plot_substitution_patterns(subst_df, output_dir):
    """Plot the top amino acid substitution patterns."""
    plt.figure(figsize=(12, 6))
    sns.barplot(x='substitution', y='count', data=subst_df)
    plt.title('Top Amino Acid Substitution Patterns')
    plt.xlabel('Substitution')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'substitution_patterns.png'), dpi=300)
    plt.close()

def plot_property_changes(property_changes, output_dir):
    """Plot the distribution of property changes."""
    properties = ['hydrophobicity', 'volume', 'polarity', 'charge', 'helix_propensity', 'sheet_propensity', 'turn_propensity']
    
    fig, axes = plt.subplots(len(properties), 1, figsize=(10, 4*len(properties)))
    
    for i, prop in enumerate(properties):
        sns.scatterplot(x='score', y=prop, data=property_changes, ax=axes[i])
        axes[i].set_title(f'Change in {prop.replace("_", " ").title()} vs. Predicted Score')
        axes[i].set_xlabel('Predicted Score')
        axes[i].set_ylabel(f'Change in {prop.replace("_", " ").title()}')
        axes[i].axhline(y=0, color='r', linestyle='--')
        
        # Add trend line
        z = np.polyfit(property_changes['score'], property_changes[prop], 1)
        p = np.poly1d(z)
        axes[i].plot(sorted(property_changes['score']), p(sorted(property_changes['score'])), "r--", alpha=0.8)
        
        # Calculate correlation
        corr = np.corrcoef(property_changes['score'], property_changes[prop])[0, 1]
        axes[i].text(0.05, 0.95, f'Correlation: {corr:.2f}', transform=axes[i].transAxes, 
                     verticalalignment='top', horizontalalignment='left',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'property_changes.png'), dpi=300)
    plt.close()

def write_analysis_report(parsed_df, top_positions, orig_aa_df, new_aa_df, subst_df, property_changes, output_dir):
    """Write an analysis report to a text file."""
    report_path = os.path.join(output_dir, 'mutation_analysis_report.md')
    
    with open(report_path, 'w') as f:
        f.write("# Mutation Analysis Report\n\n")
        
        # Summary of top positions
        f.write("## Position Analysis\n\n")
        f.write("Top positions with beneficial mutations:\n\n")
        f.write(top_positions.head(10).to_markdown(index=False))
        f.write("\n\n")
        
        # Summary of amino acid distributions
        f.write("## Amino Acid Distribution\n\n")
        f.write("### Original Amino Acids\n\n")
        f.write(orig_aa_df.head(10).to_markdown(index=False))
        f.write("\n\n")
        
        f.write("### New Amino Acids\n\n")
        f.write(new_aa_df.head(10).to_markdown(index=False))
        f.write("\n\n")
        
        # Summary of substitution patterns
        f.write("## Substitution Patterns\n\n")
        f.write("Top amino acid substitutions:\n\n")
        f.write(subst_df.to_markdown(index=False))
        f.write("\n\n")
        
        # Summary of property changes
        f.write("## Property Changes\n\n")
        f.write("Average changes in amino acid properties:\n\n")
        
        properties = ['hydrophobicity', 'volume', 'polarity', 'charge', 'helix_propensity', 'sheet_propensity', 'turn_propensity']
        prop_means = {prop: property_changes[prop].mean() for prop in properties}
        prop_df = pd.DataFrame({
            'property': list(prop_means.keys()),
            'average_change': list(prop_means.values())
        })
        f.write(prop_df.to_markdown(index=False))
        f.write("\n\n")
        
        # Correlation with score
        f.write("### Correlation with Predicted Score\n\n")
        correlations = []
        for prop in properties:
            corr = np.corrcoef(property_changes['score'], property_changes[prop])[0, 1]
            correlations.append(corr)
        
        corr_df = pd.DataFrame({
            'property': properties,
            'correlation_with_score': correlations
        })
        f.write(corr_df.to_markdown(index=False))
        f.write("\n\n")
        
        # Key findings
        f.write("## Key Findings\n\n")
        
        # Top positions
        top_pos = top_positions['position'].iloc[0]
        f.write(f"1. The most common position for beneficial mutations is {top_pos}\n")
        
        # Most common original amino acids
        top_orig_aa = orig_aa_df['amino_acid'].iloc[0]
        f.write(f"2. The most common original amino acid in beneficial mutations is {top_orig_aa}\n")
        
        # Most common new amino acids
        top_new_aa = new_aa_df['amino_acid'].iloc[0]
        f.write(f"3. The most common new amino acid in beneficial mutations is {top_new_aa}\n")
        
        # Top substitution
        top_subst = subst_df['substitution'].iloc[0]
        f.write(f"4. The most common substitution pattern is {top_subst}\n")
        
        # Property change trends
        hydrophobicity_trend = "increase" if prop_means['hydrophobicity'] > 0 else "decrease"
        volume_trend = "increase" if prop_means['volume'] > 0 else "decrease"
        helix_trend = "increase" if prop_means['helix_propensity'] > 0 else "decrease"
        sheet_trend = "increase" if prop_means['sheet_propensity'] > 0 else "decrease"
        
        f.write(f"5. Beneficial mutations tend to {hydrophobicity_trend} hydrophobicity and {volume_trend} volume\n")
        f.write(f"6. Beneficial mutations tend to {helix_trend} helix propensity and {sheet_trend} sheet propensity\n")
        
        # Property with strongest correlation
        max_corr_idx = abs(np.array(correlations)).argmax()
        max_corr_prop = properties[max_corr_idx]
        max_corr_val = correlations[max_corr_idx]
        corr_type = "positive" if max_corr_val > 0 else "negative"
        f.write(f"7. The property with the strongest correlation to mutation score is {max_corr_prop} ({corr_type} correlation, r = {max_corr_val:.2f})\n")
        
        # Check for region-specific patterns
        n_term_mutations = parsed_df[parsed_df['position'] < len(parsed_df['position'].unique()) // 3]
        mid_mutations = parsed_df[(parsed_df['position'] >= len(parsed_df['position'].unique()) // 3) & 
                                 (parsed_df['position'] < 2 * len(parsed_df['position'].unique()) // 3)]
        c_term_mutations = parsed_df[parsed_df['position'] >= 2 * len(parsed_df['position'].unique()) // 3]
        
        region_counts = {
            'N-terminal': len(n_term_mutations),
            'Middle': len(mid_mutations),
            'C-terminal': len(c_term_mutations)
        }
        
        max_region = max(region_counts, key=region_counts.get)
        f.write(f"8. Most beneficial mutations are found in the {max_region} region of the protein\n")

def main():
    # Create output directory
    output_dir = 'mutation_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load predictions
    predictions_file = 'Leaderboard/predictions.csv'
    predictions = load_predictions(predictions_file)
    
    # Filter top 1000 mutations
    top_mutations = predictions.sort_values('DMS_score_predicted', ascending=False).head(1000)
    
    # Also load the top10.txt if it exists
    top10_file = 'Leaderboard/top10.txt'
    if os.path.exists(top10_file):
        top10 = load_top_mutations(top10_file)
        print("Top 10 mutations:")
        print(top10)
    
    # Parse mutations
    parsed_df = parse_mutations_df(top_mutations)
    
    # Analyze positions
    top_positions = analyze_positions(parsed_df)
    
    # Analyze amino acid changes
    orig_aa_df, new_aa_df, subst_df = analyze_amino_acid_changes(parsed_df)
    
    # Analyze property changes
    property_changes = analyze_property_changes(parsed_df)
    
    # Generate plots
    plot_top_positions(top_positions, output_dir)
    plot_aa_distributions(orig_aa_df, new_aa_df, output_dir)
    plot_substitution_patterns(subst_df, output_dir)
    plot_property_changes(property_changes, output_dir)
    
    # Write analysis report
    write_analysis_report(parsed_df, top_positions, orig_aa_df, new_aa_df, subst_df, property_changes, output_dir)
    
    print(f"Analysis complete! Results saved to {output_dir}/")
    print(f"Report: {output_dir}/mutation_analysis_report.md")

if __name__ == "__main__":
    main() 