# Business case S&OP Analyst - Carlos Sanchez

## Demanda prediction model


The following is an explanation of the business case. **Full code explanation on the main.ipynb file.**

Requests:

1. **Pick a model and explain the reason to choose it.**
2. **Plot a time series for a specific product-customer-agencia.**
3. **Choos a metric to show the model realyabillity**
4. **Choose a dataset and explain selection criteria.**
5. **Predict Week 9 for top 3 products in the dataset.**
6. **Diagram showing all steps followed.**

## Dashboard Showing all results
Before I explain all the proceess to get a good model It is imperative for me to show you the dashboard I build for you guys to show my results.

It has 2 pages, being the second page the one where you can find the model predictions and you can access using the "Predictions" button or just by changing to page two, you can fin it in the following link:

** [Dashboard for Meli 💛](https://app.powerbi.com/view?r=eyJrIjoiYTFjNWY1ZDMtYWNlZi00NmRmLThlYWQtMjQwOTc5MWFjYjcyIiwidCI6ImY5NGJmNGQ5LTgwOTctNDc5NC1hZGY2LWE1NDY2Y2EyODU2MyIsImMiOjR9) **

![Dashboard_overview](https://github.com/The-carlos/meli_assessment/blob/main/Overview_dashboard.PNG)
![Dashboard_predictions](https://github.com/The-carlos/meli_assessment/blob/main/Predictions_dashboard.PNG)

### **TASK 1: Pick a model and explain the reason to choose it.**

This probably was the most time consuming task of all the project and I faced two principal challenges: *Error measuring* and *Computation resources consumption.* 
To handle Error measuring I just usea a simple Mean Square Error (MSE), after triying several metrics this one was easy to interpret.
Aligned with that, I tried a large variety of models, including Lass, Rigde regression, RandonForestRegressor, and more. Unfortunatelly, my local machine doesn't have enough power to handle a large variaty of models and parameters. I must use a simple LinnearRegression() desactivating the spare-matrix argumente. Only with this configuration I could create a simple pipeline to get interesting results:

```sh
# Assuming you want to use 'Venta_uni_hoy', 'Venta_hoy', 'Dev_uni_proxima', 'Dev_proxima' as features
features = ['Canal_ID', 'Ruta_SAK', 'NombreCliente', 'NombreProducto']
X = data_product_1[features]
y = data_product_1['Demanda_uni_equil']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Configure data processing using ColumnTransformer and Pipeline
preprocessor = ColumnTransformer(
    transformers = [
        ('product_customer_encoder', OneHotEncoder(handle_unknown = 'ignore', sparse = True), ['Canal_ID', 'Ruta_SAK', 'NombreCliente', 'NombreProducto'])
    ],
    remainder='passthrough'
)

#Build the pipeline model and linear regressor
model = Pipeline(steps = [
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

#Model trining
model.fit(X_train, y_train)

#Predictions
predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
print(f"Mean Square Error: {mse}")
```
The mean square error is **36.87** which is high, but not terrible and to be honest, the best result I could achive after hours of trianing process.
**There is a key concept here. I used OneHotEncoder as a transformer, this means that my model i completely able to handle unknow features not used during the training process. In other words, if there is a new product or customer, the model is able to make predictions for them.**

When plotting the real values against predicted values it's clear the model is not perfect, but is it pretty accurate:

![Model_performance](https://github.com/The-carlos/meli_assessment/blob/main/predictions_model.PNG)

### **TASK 2: Plot a time series for a specific product-customer-agencia.**

This one was pretty straight forward. You can find all details in the *main.ipynb* file, but I  it is as follows:

```sh
#TASK

#Producto: Nito 1p 62g BIM 1278
#Agencia: 1123
#Cliente: NO IDENTIFICADO

task_data = agencia_1123_data[(agencia_1123_data["NombreProducto"] == 'Nito 1p 62g BIM 1278') & 
                              (agencia_1123_data["NombreCliente"] == 'NO IDENTIFICADO')]



# Agrupar por semana y sumar las ventas en cada semana
df_agrupado = task_data.groupby('Semana')['Venta_uni_hoy'].sum().reset_index()

# Configurar el estilo de Seaborn
sns.set(style="whitegrid")

# Graficar la serie de tiempo con Seaborn
plt.figure(figsize=(10, 6))
sns.lineplot(x='Semana', y='Venta_uni_hoy', data=df_agrupado, marker='o', color='b')

# Agregar etiquetas con el valor de cada punto
for index, row in df_agrupado.iterrows():
    plt.text(row['Semana'], row['Venta_uni_hoy'], f'{row["Venta_uni_hoy"]:,.0f}', ha='center', va='bottom', fontsize=8)

plt.title('Ventas por Semana')
plt.xlabel('Semana')
plt.ylabel('Ventas')
plt.show()
```

![customer_data](https://github.com/The-carlos/meli_assessment/blob/main/specific_customer_plot.PNG)


### **TASK 3: Choose a metric to show the model realyabillity**
Already explaind in TASK 1.

### **TASK 4: Choose a dataset and explain selection criteria.**

Ok, now need to take in consideration 2 key aspects to define the Agencia to predict:

Demanda: Demanda is the most important driver for any desicion-making process. Keeping this in mind We need to have certanty of the top Agencia based on its Demanda. At first sight seem that Agencia_ID 1129,1142 and 1123 are suitable.

Number of registers available. As shown in a plot in the main.ipynb file, the Agencia with more data available is 1123. This is excellent! Also is on top 3 of Agencia with more Demanda.
**In conclusion Agencia_ID 1123 is the one with more data available and the 3rd most important in Demanda so this will be the one selected for the model.**

### **TASK 5: Predict Week 9 for top 3 products in the dataset.**
Once the modelis already trained, it was just making a iterative preduction over the top 3 products in the test data you sent me and store the results in new .csv files:

```sh
for dataset in range(len(datasets_list_to_predict)):
    test_data_predictions = model.predict(datasets_list_to_predict[dataset])
    print(f"Predictions for product {top_3_products[dataset]} done.")
    datasets_list_to_predict[dataset]["Predictions"] = test_data_predictions
    print(datasets_list_to_predict[dataset].head())
    datasets_list_to_predict[dataset].to_csv("Product_"+ str(dataset) + ".csv", index = True)


```

### **TASK 5: Diagram showing all steps followed.**
Not finished.


Thank you so much for your attention. <3
