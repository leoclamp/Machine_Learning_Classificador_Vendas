import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample

# Tira os avisos de downcasting
pd.set_option('future.no_silent_downcasting', True)

# Carregar o conjunto de dados
wholesale = pd.read_csv('Wholesale customers data.csv', delimiter=',')

# Substituir valores categóricos por numéricos
wholesale['Channel'] = wholesale['Channel'].replace({'HoReCa': 0, 'Retail': 1}).infer_objects(copy=False)
wholesale['Region'] = wholesale['Region'].replace({'Lisbon': 0, 'Oporto': 1, 'Other': 2}).infer_objects(copy=False)

# Reordenar as colunas
wholesale = wholesale.reindex(columns=['Region', 'Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicatessen', 'Channel'])

# Separar o DataFrame em características (X) e variável alvo (y)
X = wholesale.drop('Channel', axis=1)
y = wholesale['Channel']

# Balanceamento das classes (oversampling da classe minoritária)
X_majority = X[y == 0]
y_majority = y[y == 0]
X_minority = X[y == 1]
y_minority = y[y == 1]

X_minority_upsampled = resample(X_minority, 
                                replace=True, 
                                n_samples=X_majority.shape[0], 
                                random_state=42)
y_minority_upsampled = resample(y_minority, 
                                replace=True, 
                                n_samples=y_majority.shape[0], 
                                random_state=42)

X_balanced = pd.concat([X_majority, X_minority_upsampled])
y_balanced = pd.concat([y_majority, y_minority_upsampled])

# Separar o DataFrame em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Ajuste de Hiperparâmetros com GridSearchCV para k-NN
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

knn = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
knn.fit(X_train, y_train)

# Melhor modelo encontrado
best_knn = knn.best_estimator_

# Classificar as amostras do conjunto de teste
y_pred = best_knn.predict(X_test)

# Avaliar o desempenho do modelo
print("\nAcurácia do modelo:", accuracy_score(y_test, y_pred))
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))

def classify_input():
    try:
        # Obter dados do usuário com validação
        print("Insira os dados para classificação:")
        region = int(input("Region (0 para Lisbon, 1 para Oporto, 2 para Other): "))
        if region not in [0, 1, 2]:
            raise ValueError("Região inválida. Use 0, 1 ou 2.")
        
        fresh = float(input("Fresh: "))
        milk = float(input("Milk: "))
        grocery = float(input("Grocery: "))
        frozen = float(input("Frozen: "))
        detergents_paper = float(input("Detergents_Paper: "))
        delicatessen = float(input("Delicatessen: "))
        
        # Criar um DataFrame com os dados do usuário
        user_data = pd.DataFrame({
            'Region': [region],
            'Fresh': [fresh],
            'Milk': [milk],
            'Grocery': [grocery],
            'Frozen': [frozen],
            'Detergents_Paper': [detergents_paper],
            'Delicatessen': [delicatessen]
        })

        # Prever a classe com o modelo treinado
        prediction = best_knn.predict(user_data)
        
        # Mapear a previsão para os valores originais
        if prediction[0] == 0:
            print("\nO canal de vendas é: HoReCa\n")
        elif prediction[0] == 1:
            print("\nO canal de vendas é: Retail\n")
        else:
            print("C\nlasse desconhecida\n")
    except ValueError as e:
        print(f"\nErro de entrada: {e}\n")
    except Exception as e:
        print(f"\nOcorreu um erro: {e}\n")

# Chamar a função para obter e classificar a entrada do usuário
classify_input()

