# Sberbank Russian Housing Market Kaggle Challenge

Description : Housing costs demand a significant investment from both consumers and developers. And when it comes to planning a budget—whether personal or corporate—the last thing anyone needs is uncertainty about one of their biggets expenses. [Sberbank](https://www.kaggle.com/sberbank), Russia’s oldest and largest bank, helps their customers by making predictions about realty prices so renters, developers, and lenders are more confident when they sign a lease or purchase a building.

Although the housing market is relatively stable in Russia, the country’s volatile economy makes forecasting prices as a function of apartment characteristics a unique challenge. Complex interactions between housing features such as number of bedrooms and location are enough to make pricing predictions complicated. Adding an unstable economy to the mix means Sberbank and their customers need more than simple regression models in their arsenal.

In this competition, Sberbank is challenging Kagglers to develop algorithms which use a broad spectrum of features to predict realty prices. Competitors will rely on a rich dataset that includes housing data and macroeconomic patterns. An accurate forecasting model will allow Sberbank to provide more certainty to their customers in an uncertain economy.

## Setting up environment 

Create a virtual environment and activate it.
    
    virtualenv env
    source env/bin/activate
    
Download the files from Kaggle and move them to the Data.

    chmod +x extract.sh
    ./extract.sh
    
Install the requirements.
    
    pip install -r requirements.txt
