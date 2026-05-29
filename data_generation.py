import pandas as pd
import numpy as np

# Configuração de reprodutibilidade
np.random.seed(42)
num_samples = 1000

# Divisão das classes
classes = np.random.choice(['Satelite', 'Lixo Espacial', 'Asteroide'], size=num_samples, p=[0.3, 0.5, 0.2])

data = []

for obj_class in classes:
    if obj_class == 'Satelite':
        rcs = np.random.normal(5.0, 1.5) # Tamanho médio
        albedo = np.random.uniform(0.4, 0.8) # Alto reflexo
        vel = np.random.normal(7.5, 0.5) # Velocidade orbital típica LEO
        ecc = np.random.uniform(0.0001, 0.05) # Órbita quase circular
        rf = np.random.choice([1, 0], p=[0.95, 0.05]) # Quase sempre emite sinal
        spin = np.random.normal(0, 0.5) # Estabilizado
        temp = np.random.normal(290, 10) # Temperatura controlada
        inc = np.random.uniform(0, 100)
        brilho_var = np.random.uniform(1, 5) # Brilho constante
        
    elif obj_class == 'Lixo Espacial':
        rcs = np.abs(np.random.normal(0.5, 0.8)) # Geralmente pedaços pequenos
        albedo = np.random.uniform(0.1, 0.6) # Reflexo variável
        vel = np.random.normal(7.5, 1.0) # Velocidade similar a satélites
        ecc = np.random.uniform(0.0001, 0.1) # Pode ser levemente elíptica
        rf = 0 # Sem sinal
        spin = np.abs(np.random.normal(15, 10)) # Capotamento descontrolado (alta rotação)
        temp = np.random.normal(200, 50) # Frio/variável, sem controle térmico
        inc = np.random.uniform(0, 100)
        brilho_var = np.random.uniform(20, 80) # Variação brusca de brilho ao capotar
        
    else: # Asteroide
        rcs = np.abs(np.random.normal(50, 100)) # Podem ser enormes
        albedo = np.random.uniform(0.05, 0.2) # Muito escuros (rocha)
        vel = np.random.normal(20, 10) # Muito rápidos, trajetórias cruzadas
        ecc = np.random.uniform(0.2, 0.9) # Órbitas altamente excêntricas
        rf = 0 # Sem sinal
        spin = np.random.uniform(0.1, 5) # Rotação lenta
        temp = np.random.normal(150, 40) # Frios
        inc = np.random.uniform(0, 180)
        brilho_var = np.random.uniform(5, 15) # Variação suave dependendo do formato
        
    data.append([
        max(0.01, rcs), albedo, abs(vel), ecc, rf, 
        abs(spin), temp, inc, brilho_var, obj_class
    ])

# Criando o DataFrame
columns = [
    'rcs_m2', 'albedo', 'velocidade_relativa_km_s', 'excentricidade_orbital',
    'emissao_sinal_rf', 'taxa_rotacao_rpm', 'assinatura_termica_k',
    'inclinacao_orbital_graus', 'variacao_brilho_porcentagem', 'classe'
]

df = pd.DataFrame(data, columns=columns)

# Salvando em CSV
df.to_csv('synthetic_space_objects.csv', index=False)
print("Banco de dados gerado com sucesso com", len(df), "linhas.")
print(df.head())