# Instalação
# !pip install pycaret

import pandas as pd
from pycaret.classification import *

# 1) Carregar dados
df = pd.read_csv('./synthetic_space_objects.csv')

# 2) Iniciar experimento AutoML
exp = setup(
    data=df,
    target='classe',
    session_id=42,
    train_size=0.7,
    normalize=True,
    fold=5,
    verbose=True
)

# 3) Comparar modelos
best_model = compare_models()

# 4) Ver leaderboard completo
leaderboard = pull()
print(leaderboard.head(10))

# 5) Ajustar o melhor modelo
tuned_model = tune_model(best_model)

# 6) Avaliar visualmente
plot_model(tuned_model, plot='residuals')
plot_model(tuned_model, plot='error')
plot_model(tuned_model, plot='feature')

# 7) Finalizar modelo
final_model = finalize_model(tuned_model)

# 8) Previsões
predictions = predict_model(final_model)
print(predictions.head())

# 9) Salvar modelo
save_model(final_model, './pycaret_space_objects_model')