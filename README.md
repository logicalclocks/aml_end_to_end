# AMLend2end
AML End to End Example


## Notebooks description

#### SPR_ingestion.ipynb
Create 2 features groups from the Santanted Product Recommendation competition https://www.kaggle.com/c/santander-product-recommendation/data .
The feature groups are one for train and the other for test.

#### AMLSim_injestion.ipynb
Create 2 features groups from the AMLSim project https://github.com/IBM/AMLSim .
The feature groups represents the generated accounts and transactions.

#### feature_engineering.ipynb
Create 2 feature groups from the feature groups created in AMLSim_injestion.ipynb.
The features are transformed in numerical.
Finally, the notebook creates 2 training datasets frin the feature groups.


## Data description

#### demodata/accounts.csv
Generated accounts from AMLSim integrated with gender and age data.

#### demodata/transactions.csv
Generated transactions from AMLSim.

#### demodata/transactions_graph.csv
Generated alert transactions graph.
