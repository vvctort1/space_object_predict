# ARQUIVO UTILIZADO NO UPLOAD DO HUGGING FACE SPACE PARA DEPLOY DO MODELO XGBOOST DE CLASSIFICAÇÃO DE OBJETOS ESPACIAIS

import pandas as pd
import numpy as numpy
import joblib
import gradio as gr



modelo_xgb_carregado = joblib.load("./src/models/modelo_xgb.joblib")


mapeamento = {
    0: "Lixo Espacial",
    1: "Satélite Ativo",
    2: "Asteroide"
}

def predict_price(Rcs_m2, AlbedoOptico, TaxaRotacaoRPM, VariacaoTermicaK, 
                  DesvioTrajetoriaPrevistaM, IndiceEspectralMetal, 
                  RugosidadeSuperficie, DensidadeEstimadaKgM3, VariacaoBrilhoLuzSolar):

    input_data = pd.DataFrame([{
        'rcs_m2': Rcs_m2,                                    
        'albedo_optico': AlbedoOptico,                        
        'taxa_rotacao_rpm': TaxaRotacaoRPM,                   
        'variacao_termica_k': VariacaoTermicaK,                
        'desvio_trajetoria_prevista_m': DesvioTrajetoriaPrevistaM,
        'indice_espectral_metal': IndiceEspectralMetal,
        'rugosidade_superficie': RugosidadeSuperficie,
        'densidade_estimada_kg_m3': DensidadeEstimadaKgM3,
        'variacao_brilho_luz_solar': VariacaoBrilhoLuzSolar
    }])

    prediction = modelo_xgb_carregado.predict(input_data)[0]
    return f"{mapeamento[int(prediction)]}".upper()


with gr.Blocks(title="Detecção de Objetos Espaciais - XGBoost") as app:

    gr.Markdown("""
    # Predição de Objetos Espaciais com XGBoost
    Modelo **XGBoost Classifier** treinado com o dataset **objetos_espaciais_complexo.csv**.
    """)

    gr.Markdown("""
    ### Legenda das Features

    | Coluna | Significado |
    |---|---|
    | `rcs_m2` | Radar Cross Section — área de reflexão do objeto no radar, em m² |
    | `albedo_optico` | Fração da luz solar refletida pelo objeto (0 = absorve tudo, 1 = reflete tudo) |
    | `taxa_rotacao_rpm` | Velocidade de rotação do objeto em torno do próprio eixo, em RPM |
    | `variacao_termica_k` | Variação de temperatura superficial do objeto, em Kelvin |
    | `desvio_trajetoria_prevista_m` | Diferença entre a trajetória real e a prevista, em metros |
    | `indice_espectral_metal` | Índice que indica a composição metálica com base na assinatura espectral |
    | `rugosidade_superficie` | Irregularidade da superfície do objeto (quanto maior, mais rugosa) |
    | `densidade_estimada_kg_m3` | Densidade estimada do objeto, em kg/m³ |
    | `variacao_brilho_luz_solar` | Variação na intensidade da luz solar refletida ao longo do tempo |
    | `tipo_objeto` | Classe: **0** = Lixo Espacial · **1** = Satélite Ativo · **2** = Asteroide |
    """)


    with gr.Row():
        with gr.Column():
            Rcs_m2 = gr.Number(label="Rcs_m2 - Área de radar (m²)", value=31.260238)
            AlbedoOptico = gr.Number(label="Albedo óptico (0-1)", value=0.199491)
            TaxaRotacaoRPM = gr.Number(label="Taxa de rotação (RPM)", value=1.690157)

        with gr.Column():
            VariacaoTermicaK = gr.Number(label="Variação térmica (K)", value=90.867466)
            DesvioTrajetoriaPrevistaM = gr.Number(label="Desvio da trajetória prevista (m)", value=4.907679)
            IndiceEspectralMetal = gr.Number(label="Índice espectral de metal", value=0.068184)

        with gr.Column():
            RugosidadeSuperficie = gr.Number(label="Rugosidade da superfície (0-1)", value=0.976906)
            DensidadeEstimadaKgM3 = gr.Number(label="Densidade estimada (kg/m³)", value=2032.525479)
            VariacaoBrilhoLuzSolar = gr.Number(label="Variação do brilho da luz solar (0-1)", value=0.213603)

    output = gr.Textbox(
        label="Objeto Detectado",
        lines=2,
        text_align="center",
    )

    with gr.Row():
        predict_button = gr.Button("Fazer predição", variant="primary")
        clear_button = gr.Button("Limpar campos", variant="secondary")

    predict_button.click(
        fn=predict_price,
        inputs=[
            Rcs_m2, AlbedoOptico, TaxaRotacaoRPM, VariacaoTermicaK, DesvioTrajetoriaPrevistaM, IndiceEspectralMetal, RugosidadeSuperficie, DensidadeEstimadaKgM3, VariacaoBrilhoLuzSolar
        ],
        outputs=output
    )

    clear_button.click(
        fn=lambda: [None, None, None, None, None, None, None, None, None, ""],
        inputs=[],
        outputs=[
            Rcs_m2, AlbedoOptico, TaxaRotacaoRPM, VariacaoTermicaK, DesvioTrajetoriaPrevistaM, IndiceEspectralMetal, RugosidadeSuperficie, DensidadeEstimadaKgM3, VariacaoBrilhoLuzSolar, output
        ]
    )

    gr.Examples(
        examples=[
            [31.260238, 0.199491, 1.690157, 90.867466, 4.907679, 0.068184, 0.976906, 2032.525479, 0.213603],
            [2.726820, 0.480800, 30.185397, 115.546072, 14.368404, 0.408652, 0.290703, 2354.226490, 0.417699],
            [6.962419, 0.831294, 0.608982, 23.282290, 3.066657, 1.003771, 0.146154, 1486.924177, 0.380237]
        ],
        inputs=[
            Rcs_m2, AlbedoOptico, TaxaRotacaoRPM, VariacaoTermicaK, DesvioTrajetoriaPrevistaM, IndiceEspectralMetal, RugosidadeSuperficie, DensidadeEstimadaKgM3, VariacaoBrilhoLuzSolar
        ]
    )

app.launch(share=True)

