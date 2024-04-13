import dash
from dash import dcc, html
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

app = dash.Dash(__name__)

port_stem = PorterStemmer()
try:
    vector_form = pickle.load(open('vector.pkl', 'rb'))
    load_model = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    print("Wystąpił błąd podczas ładowania modelu lub transformatora TF-IDF:", e)

def stemming(content):
    con = re.sub('[^a-zA-Z]', ' ', content)
    con = con.lower()
    con = con.split()
    con = [port_stem.stem(word) for word in con if not word in stopwords.words('english')]
    con = ' '.join(con)
    return con


def check_news(news):
    news = stemming(news)
    input_data = [news]
    vector_form1 = vector_form.transform(input_data)
    prediction = load_model.predict(vector_form1)
    return prediction


app.layout = html.Div(
    style={
        'display': 'flex',
        'align-items': 'center',
        'justify-content': 'center',
        'height': '100vh',
        'background-color': 'black',
        'font-family': 'Arial, sans-serif'
    },
    children=[
        html.Div(
            style={
                'background-color': 'rgba(255, 255, 255, 0.1)',
                'border-radius': '10px',
                'padding': '20px',
                'width': '80%',
                'text-align': 'center'
            },
            children=[
                html.H1('Weryfikacja wiarygodności artykułu', 
                style={
                    'color': 'white', 
                    'padding-top': '20px'}),
                html.Div([
                    html.H3('Wpisz treść twojego artykułu do weryfikacji', 
                    style={
                    'color': 'white', 
                    'text-align': 'left', 
                    'margin-left': '20px'}),
                    dcc.Textarea(
                        id='textarea-input',
                        value='',
                        placeholder='Wprowadź tekst artykułu',
                        style={
                        'width': '80%', 
                        'height': '200px', 
                        'margin': '20px auto', 
                        'border-radius': '10px', 
                        'padding': '10px'}
                    ),
                    html.Button(
                        'Predykcja',
                        id='predict-button',
                        n_clicks=0,
                        style={
                        'display': 'block', 
                        'margin': '0 auto', 
                        'border-radius': '5px', 
                        'padding': '15px 30px', 
                        'background-color': 'gray',
                        'color': 'white'}
                    )
                ]),
                html.Div(id='prediction-output', 
                style={
                    'margin-top': '20px',
                    'text-align': 'center'})
            ]
        )
    ]
)

@app.callback(
    dash.dependencies.Output('prediction-output', 'children'),
    [dash.dependencies.Input('predict-button', 'n_clicks')],
    [dash.dependencies.State('textarea-input', 'value')]
)

def update_output(n_clicks, value):

    if n_clicks > 0:
        if value is None or value.strip() == '':
            return html.Div( 'Najpierw wprowadź tekst',
                style={
                    'color': 'white',
                    'background-color': 'gray',
                    'border-radius': '10px',
                    'padding': '20px',
                    'width': '50%',
                    'margin': '20px auto'}
            )
        else:
            prediction_class = check_news(value)
            if prediction_class == [0]:
                return html.Div( 'Wiarygodny',
                    style={
                        'color': 'white', 
                        'background-color': 'green', 
                        'border-radius': '10px', 
                        'padding': '20px', 
                        'width': '50%', 
                        'margin': '20px auto'}
                )
            elif prediction_class == [1]:
                return html.Div( 'Niewiarygodny',
                    style={'color': 'white', 
                    'background-color': 'red', 
                    'border-radius': '10px', 
                    'padding': '20px', 
                    'width': '50%', 
                    'margin': '20px auto'}
                )

if __name__ == '__main__':
    app.run_server(debug=True)

