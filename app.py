import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import numpy as np

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Set page config
st.set_page_config(page_title="Análise de Reclamações", layout="wide")


# Function to load and preprocess data
@st.cache_data
def load_data():
    # Replace this with your actual data loading logic
    df = pd.read_csv('Trust Works_ Reclamações Corporativas Anonimizadas - Reclamações.csv', header=13, index_col=0)
    df.index = pd.to_datetime(df.index, format='mixed')
    df.sort_index(inplace=True)
    return df.iloc[:3000].dropna()

# Function to preprocess text
def preprocess_text(text):
    if pd.isna(text):
        return ""
    # Convert to lowercase
    text = str(text).lower()
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Truncate to max 512 tokens for sentiment analysis
    return ' '.join(text.split()[:512])



# Modified Topic Modeling function with correct document visualization
@st.cache_resource
def perform_topic_modeling(texts):
    if len(texts) < 50:
        raise ValueError("Insufficient data for topic modeling. Need at least 50 documents.")
        
    embedding_model = SentenceTransformer('ricardo-filho/bert-base-portuguese-cased-nli-assin-2')
    
    topic_model = BERTopic(
        embedding_model=embedding_model,
        nr_topics=min(20, len(texts) // 50),
        min_topic_size=5,
        top_n_words=10,
        verbose=True
    )
    
    topics, _ = topic_model.fit_transform(texts)
    
    if len(topic_model.get_topic_info()) <= 1:
        raise ValueError("No meaningful topics were found in the data.")
        
    return topic_model, topics

# Modified sentiment analysis function
@st.cache_resource
def load_sentiment_model():
    model_name = "neuralmind/bert-base-portuguese-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)  # Ensure we have multiple labels
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, max_length=512, truncation=True)

# Load data
df = load_data()

# Sidebar
st.sidebar.title("Filtros")

# Company selection
all_companies = ['Todas as Empresas'] + sorted(df['Empresa'].unique().tolist())
selected_company = st.sidebar.selectbox(
    "Selecione a Empresa",
    all_companies
)

# Date range selection
date_range = st.sidebar.date_input(
    "Selecione o período",
    [df.index.min(), df.index.max()],
    min_value=df.index.min().date(),
    max_value=df.index.max().date()
)

# Filter data based on date range and company
mask = (df.index.date >= date_range[0]) & (df.index.date <= date_range[1])
filtered_df = df[mask]

if selected_company != 'Todas as Empresas':
    filtered_df = filtered_df[filtered_df['Empresa'] == selected_company]

# Main content
st.title("Dashboard de Análise de Reclamações")

st.markdown("""
            ## Feito por [Felipe Gabriel](https://www.linkedin.com/in/felipe-gabriel0/)

            ### Fonte dos dados
            
            https://docs.google.com/spreadsheets/d/1aTxJq6PMKckLhRQVA6ZdcG3QjMGVZ_QHbjun3Ylq_JA/edit?gid=0#gid=0""")

# Top companies by complaints and cumulative complaints

# Main content with company visualization
col1, col2 = st.columns(2)

with col1:
    st.subheader("Empresas mais Reclamadas")
    top_companies = filtered_df['Empresa'].value_counts().head(10)
    fig_companies = px.bar(
        top_companies,
        title="Top 10 Empresas com Mais Reclamações",
        labels={'value': 'Número de Reclamações', 'index': 'Empresa'}
    )
    st.plotly_chart(fig_companies, use_container_width=True)
    
    st.markdown("""
    **Como interpretar este gráfico:**
    - As barras mostram as 10 empresas com maior número de reclamações
    - A altura de cada barra representa o número total de reclamações
    - Empresas são ordenadas da mais reclamada (esquerda) para a menos reclamada (direita)
    - Use este gráfico para identificar as empresas que requerem mais atenção em termos de satisfação do cliente
    """)

with col2:
    st.subheader("Evolução das Reclamações")
    
    daily_complaints = filtered_df.resample('D').size()
    cumulative_complaints = daily_complaints.cumsum()
    
    fig_cumulative = go.Figure()
    
    fig_cumulative.add_trace(
        go.Scatter(
            x=cumulative_complaints.index,
            y=cumulative_complaints.values,
            mode='lines',
            name='Reclamações Acumuladas',
            line=dict(color='blue')
        )
    )
    
    fig_cumulative.add_trace(
        go.Bar(
            x=daily_complaints.index,
            y=daily_complaints.values,
            name='Reclamações Diárias',
            opacity=0.3
        )
    )
    
    fig_cumulative.update_layout(
        title=f"Evolução das Reclamações {': ' + selected_company if selected_company != 'Todas as Empresas' else ''}",
        xaxis_title="Data",
        yaxis_title="Número de Reclamações",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_cumulative, use_container_width=True)
    
    st.markdown("""
    **Como interpretar este gráfico:**
    - Barras cinzas: Mostram o número de reclamações recebidas por dia
    - Linha azul: Mostra o total acumulado de reclamações ao longo do tempo
    - Use este gráfico para identificar:
        - Tendências de aumento ou diminuição no volume de reclamações
        - Dias com picos anormais de reclamações
        - Crescimento geral do volume de reclamações
    """)



# NLP Analysis
st.header("Análise de Texto das Reclamações")

if st.button("Realizar Análise Completa"):
    with st.spinner("Processando análises..."):
        preprocessed_complaints = [preprocess_text(text) for text in filtered_df['Reclamação']]
        preprocessed_complaints = [text for text in preprocessed_complaints if len(text.split()) >= 5]
        
        try:
            if len(preprocessed_complaints) >= 50:
                topic_model, topics = perform_topic_modeling(preprocessed_complaints)
                
                tab1, tab2, tab3 = st.tabs(["Tópicos Principais", "Visualização de Documentos", "Palavras Frequentes"])
                
                with tab1:
                    st.subheader("Principais Tópicos Identificados")
                    topic_info = topic_model.get_topic_info()
                    relevant_topics = topic_info[topic_info['Topic'] != -1].head(5)
                    
                    topic_display = []
                    for _, row in relevant_topics.iterrows():
                        topic_words = topic_model.get_topic(row['Topic'])
                        words = [word for word, _ in topic_words[:5]]
                        topic_display.append({
                            'Tópico': f"Tópico {row['Topic']}",
                            'Palavras-chave': ', '.join(words),
                            'Quantidade': row['Count']
                        })
                    
                    st.dataframe(pd.DataFrame(topic_display))
                    
                    fig_topics = topic_model.visualize_topics()
                    st.plotly_chart(fig_topics, use_container_width=True)
                    
                    st.markdown("""
                    **Como interpretar esta análise de tópicos:**
                    - Cada tópico representa um grupo de reclamações com conteúdo similar
                    - 'Palavras-chave' mostra os termos mais característicos de cada tópico
                    - 'Quantidade' indica quantas reclamações pertencem a cada tópico
                    - No gráfico interativo, pontos próximos indicam tópicos relacionados
                    """)
                
                with tab2:
                    st.subheader("Visualização de Documentos e Sentimentos")
                    
                    # Document Visualization
                    try:
                        doc_viz = topic_model.visualize_documents(
                            preprocessed_complaints,
                            topics=topics,
                            #reduce_function='UMAP'
                        )
                        st.plotly_chart(doc_viz, use_container_width=True)
                        
                        st.markdown("""
                        **Como interpretar a visualização de documentos:**
                        - Cada ponto representa uma reclamação individual
                        - Cores diferentes indicam diferentes tópicos
                        - Pontos próximos representam reclamações com conteúdo similar
                        - Agrupamentos (clusters) sugerem padrões comuns nas reclamações
                        """)
                    except Exception as e:
                        st.error(f"Erro na visualização de documentos: {str(e)}")
                    
                    # Sentiment Analysis
                    st.subheader("Análise de Sentimentos")
                    try:
                        sentiment_analyzer = load_sentiment_model()
                        
                        sample_size = min(100, len(filtered_df))
                        sample_complaints = filtered_df['Reclamação'].sample(n=sample_size)
                        
                        sample_texts = [preprocess_text(text) for text in sample_complaints]
                        sample_texts = [text for text in sample_texts if text.strip()]
                        
                        sentiments = []
                        for text in sample_texts:
                            try:
                                result = sentiment_analyzer(text)[0]
                                sentiments.append(result)
                            except Exception as e:
                                continue
                        
                        if sentiments:
                            sentiment_counts = Counter([s['label'] for s in sentiments])
                            total = sum(sentiment_counts.values())
                            
                            sentiment_percentages = {
                                label: (count / total) * 100 
                                for label, count in sentiment_counts.items()
                            }
                            
                            # Better labels for sentiment categories
                            sentiment_labels = {
                                'LABEL_0': 'Positivo',
                                'LABEL_1': 'Neutro',
                                'LABEL_2': 'Negativo'
                            }
                            
                            renamed_sentiments = {
                                sentiment_labels.get(k, k): v 
                                for k, v in sentiment_percentages.items()
                            }
                            
                            fig_sentiment = px.pie(
                                values=list(renamed_sentiments.values()),
                                names=list(renamed_sentiments.keys()),
                                title="Distribuição de Sentimentos nas Reclamações",
                                color_discrete_sequence=['#2ecc71', '#3498db', '#e74c3c']  # Green, Blue, Red
                            )
                            
                            fig_sentiment.update_traces(textposition='inside', textinfo='percent+label')
                            
                            st.plotly_chart(fig_sentiment, use_container_width=True)
                            
                            st.markdown("""
                            **Como interpretar a análise de sentimentos:**
                            - Verde (Positivo): Reclamações com tom mais construtivo ou resolução satisfatória
                            - Azul (Neutro): Reclamações com tom mais factual ou descritivo
                            - Vermelho (Negativo): Reclamações com tom mais crítico ou insatisfação explícita
                            
                            **Detalhamento dos sentimentos:**
                            """)
                            
                            for label, count in sentiment_counts.items():
                                st.write(f"{sentiment_labels[label]}: {count} reclamações ({(count/total)*100:.1f}%)")
                            
                            st.markdown("""
                            **Observações importantes:**
                            - Esta análise é baseada em uma amostra aleatória de até 100 reclamações
                            - O tom da reclamação não necessariamente reflete a gravidade do problema
                            - Use esta análise para identificar tendências gerais no tom das reclamações
                            """)
                        else:
                            st.warning("Não foi possível analisar os sentimentos nos textos fornecidos.")
                            
                    except Exception as e:
                        st.error(f"Erro na análise de sentimento: {str(e)}")
                
                with tab3:
                    st.subheader("Análise de Frequência de Palavras")
                    try:
                        stop_words = set(stopwords.words('portuguese'))
                        
                        all_words = []
                        for complaint in preprocessed_complaints:
                            words = word_tokenize(complaint)
                            words = [w for w in words if w not in stop_words and len(w) > 2]
                            all_words.extend(words)
                        
                        word_freq = Counter(all_words).most_common(20)
                        
                        fig_words = px.bar(
                            x=[w[0] for w in word_freq],
                            y=[w[1] for w in word_freq],
                            title="Palavras Mais Frequentes nas Reclamações",
                            labels={'x': 'Palavra', 'y': 'Frequência'}
                        )
                        st.plotly_chart(fig_words, use_container_width=True)
                        
                        st.markdown("""
                        **Como interpretar o gráfico de frequência de palavras:**
                        - Barras mostram as palavras mais utilizadas nas reclamações
                        - Altura da barra indica quantas vezes a palavra aparece
                        - Palavras comuns (stop words) foram removidas
                        - Use este gráfico para identificar termos recorrentes e problemas frequentes
                        """)
                    except Exception as e:
                        st.error(f"Erro na análise de frequência de palavras: {str(e)}")
            else:
                st.warning("Dados insuficientes para análise de tópicos. São necessários pelo menos 50 documentos.")
                
        except Exception as e:
            st.error(f"Erro na análise de tópicos: {str(e)}")

# Add export functionality
if st.button("Exportar Análise"):
    # Create a summary DataFrame
    summary = pd.DataFrame({
        'Total de Reclamações': [len(filtered_df)],
        'Número de Empresas': [filtered_df['Empresa'].nunique()],
        'Período da Análise': [f"{filtered_df.index.min().date()} até {filtered_df.index.max().date()}"],
        'Empresa Analisada': [selected_company]
    })
    
    # Convert to CSV
    csv = summary.to_csv(index=False)
    st.download_button(
        label="Download Resumo CSV",
        data=csv,
        file_name="analise_reclamacoes.csv",
        mime="text/csv"
    )
