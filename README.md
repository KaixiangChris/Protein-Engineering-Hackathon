# Protein Engineering Optimization Project

This repository contains improved machine learning models for protein engineering through mutation effect prediction.

## Project Summary

We have developed a suite of optimized models that can accurately predict the effect of amino acid mutations on protein stability and function. Our best model achieves a Spearman correlation of 0.52 on the validation set, indicating strong predictive performance.

## Key Achievements

1. **Simple Optimized SVR Model**: We built a streamlined SVR model that utilizes carefully engineered features based on amino acid properties, position information, and substitution patterns. The model was optimized using RandomizedSearchCV to find the best hyperparameters.

2. **Comprehensive Feature Engineering**: Our approach incorporated over 70 engineered features including:
   - Amino acid substitution patterns (BLOSUM62 scores)
   - Physicochemical property changes (hydrophobicity, volume, charge, etc.)
   - Position-specific information
   - Secondary structure propensities

3. **Mutation Analysis**: We conducted an in-depth analysis of beneficial mutations, revealing valuable insights for protein engineering.

## Model Performance

Our simple optimized SVR model achieved:
- **Validation Spearman Correlation**: 0.52
- **Best Hyperparameters**: 
  - Kernel: linear
  - C: 10.0
  - Gamma: 1.0
  - Epsilon: 0.1

## Mutation Analysis Insights

### Top Predicted Beneficial Mutations

The top 10 predicted beneficial mutations are:
```
R650K (2.69)
R650Q (2.64)
R644K (2.63)
N648H (2.60)
N648D (2.59)
R644Q (2.58)
R650H (2.57)
R650E (2.57)
V635I (2.57)
N648Q (2.56)
```

### Key Patterns Identified

1. **Position Hotspots**: The most beneficial mutations are concentrated at positions 649, 643, 647, 634, and 654, suggesting critical functional regions in the protein.

2. **Amino Acid Preferences**:
   - Leucine (L) is the most common original amino acid in beneficial mutations
   - Methionine (M) is the most common substitution residue

3. **Substitution Patterns**: The most frequent beneficial substitutions are L→M and L→I, indicating a preference for conservative hydrophobic substitutions.

4. **Property Changes**:
   - Beneficial mutations tend to slightly decrease hydrophobicity and volume
   - They tend to increase sheet propensity while decreasing helix propensity
   - The strongest correlation with mutation score is sheet propensity (r = 0.14)

5. **Regional Preference**: Most beneficial mutations are located in the C-terminal region of the protein, indicating this region's importance for stability or function.

## Biological Implications

The concentration of beneficial mutations in the C-terminal region (positions ~635-650) suggests this region plays a critical role in the protein's function or stability. The preference for substitutions that decrease hydrophobicity while increasing sheet propensity may indicate that these changes enhance the formation of stable secondary structures.

The prevalence of arginine (R) and asparagine (N) as original amino acids in the top 10 mutations, often substituted with lysine (K), histidine (H), or aspartic acid (D), suggests that modulating charge or hydrogen bonding capabilities in this region is important. The R→K substitutions represent conservative changes that maintain positive charge but may provide more conformational flexibility.

## Usage

To run the optimized SVR model:
```
python simple_optimized_svr.py --data_dir Hackathon_data --select_features 100
```

To analyze the mutation patterns:
```
python analyze_mutations.py
```

## Directory Structure

- `Hackathon_data/`: Contains the original dataset
- `simple_optimized_svr_models/`: Contains the trained model files
- `simple_optimized_svr_results/`: Contains the model predictions and results
- `mutation_analysis/`: Contains the mutation analysis report and visualizations
- `Leaderboard/`: Contains the top predicted mutations

## Requirements

- Python 3.8+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

## Future Directions

1. **Deep Learning Integration**: Explore integration of protein language models like ESM for better feature representation.
2. **Structural Considerations**: Incorporate 3D structural information for more accurate predictions.
3. **Ensemble Methods**: Develop more sophisticated ensemble approaches combining multiple model types.
4. **Focused Experimentation**: Test the top predicted mutations experimentally, focusing on the C-terminal region. 