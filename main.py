import pandas as pd
from cleaner.cleaner import Cleaner
import streamlit as st
import requests
import json

API_ENDPOINT = 'https://ml-fastapi-liao.onrender.com/predict'

__type_subtype = {"house": ('chalet', 'bungalow', 'town-house', 'villa', 'castle', 'farmhouse', 'mansion', 'mixed-use-building', 'country-cottage', 'manor-house', 'house', 'other-property', 'exceptional-property', 'apartment-block'), 
                    "apartment": ('flat-studio', 'loft', 'service-flat', 'duplex', 'triplex', 'apartment', 'penthouse', 'kot', 'ground-floor')}
numerical_features = ["bedroomCount", "habitableSurface", "facedeCount", "streetFacadeWidth", "kitchenSurface", "landSurface", 
                    "terraceSurface", "gardenSurface", "toiletCount", "bathroomCount"]
categorical_features = [
    "type",
    "subtype",
    "province",
    "locality",
    "postCode",
    "hasBasement",
    "buildingCondition",
    "buildingConstructionYear",
    "hasTerrace",
    "floodZoneType",
    "heatingType",
    "kitchenType",
    "gardenOrientation",
    "hasSwimmingPool",
    
    "terraceOrientation",
    "epcScore"
]

postCodes_df = pd.read_csv("./data/georef-belgium-postal-codes.csv", 
                           sep=";", 
                           usecols = ['Post code', 'Région name (French)', 'Province name (French)',
                                      'Arrondissement name (French)', 'Municipality name (Dutch)',
                                      'Municipality name (French)', 'Sub-municipality name (French)',
                                      'Sub-municipality name (Dutch)'], header = 0).rename(
                                          columns = {'Post code': 'postCode', 'Région name (French)': 'rgn_fr',
                                                     'Province name (French)': 'prov_fr', 'Arrondissement name (French)' : 'arrond_fr',
                                                     'Municipality name (French)': 'mun_fr', 'Municipality name (Dutch)': 'mun_nl',
                                                     'Sub-municipality name (French)': 'sub_mun_fr', 'Sub-municipality name (Dutch)': 'sub_mun_nl'})
def get_values_from_df(column_name, df) :
    """
    Get the list of unique values in a column of a dataframe.
    """
    return list(filter(lambda x: isinstance(x, str), df[column_name].sort_values().unique()))

def main() :
    df = Cleaner.clean_data()
 
    df = df.merge(postCodes_df, how = 'inner', on = 'postCode')
    df['municipality'] = df['mun_fr'].combine_first(df['mun_nl'])
    df['sub_municipality'] = df['sub_mun_fr'].combine_first(df['sub_mun_nl'])
        
    st.title("House Price Prediction")
    st.write("This is a simple house price prediction app.")
    st.write("Please fill in the form below to get a prediction.")

    options = ["APARTMENT", "HOUSE"]
    tp = st.pills("Type of property", options, selection_mode="single", default = "APARTMENT")
    ind = 0
    if tp == "HOUSE" :
        ind = 7 # index of default value HOUSE in subtypes
    subtypes = list(map(lambda x: x.upper().replace('-', '_'), list(__type_subtype[tp.lower()])))
    subtypes.sort()
    subtype = st.selectbox('Subtype', subtypes, index = ind)
    
    region = st.selectbox('Region', get_values_from_df('rgn_fr', df), index = None)
    provinces = get_values_from_df('prov_fr', df[df.rgn_fr == region])
    province = ''
    if len(provinces) > 0 :
        province = st.selectbox('Province', provinces, index = None)
    arrond = st.selectbox('Arrondissement', get_values_from_df('arrond_fr', df[(df.rgn_fr == region) & (len(provinces) == 0 or df.prov_fr == province)]), index = None)
    municipality = st.selectbox('Municipality', get_values_from_df('municipality', df[(df.rgn_fr == region) & (len(provinces) == 0 or df.prov_fr == province) & (df.arrond_fr == arrond)]), index = None)
    sub_muns = get_values_from_df('sub_municipality', df[(df.rgn_fr == region) & (len(provinces) == 0 or df.prov_fr == province) & (df.arrond_fr == arrond) & (df.municipality == municipality)])
    subMunicipality = ''
    if len(sub_muns) > 1 :
        subMunicipality = st.selectbox('Sub-municipality', sub_muns, index = None)
    elif len(sub_muns) == 1 :
        subMunicipality = sub_muns[0]
    
    postCodes = df[(df.rgn_fr == region) & (len(provinces) == 0 or df.prov_fr == province) & (df.arrond_fr == arrond) & (df.municipality == municipality) & (len(sub_muns) == 0 or df.sub_municipality == subMunicipality)]['postCode'].sort_values().unique()
    if len(postCodes) > 1 :
        postCode = st.selectbox('Post code', postCodes)
    elif len(postCodes) == 1 :
        postCode = postCodes[0]

    conditions = get_values_from_df('buildingCondition', df)
    buildingCondition = st.selectbox('Building condition', conditions, index=None)
    for col in filter(lambda x: x.startswith('has'), df[numerical_features + categorical_features].columns.to_list()) :
        st.checkbox(f'Has {col[3:].lower()}', key = col)
    buildingConstructionYear = st.number_input('Building construction year', min_value = 1900, max_value = 2025, value = 2024)
    heatingType = st.selectbox('Heating type', get_values_from_df('heatingType', df), index=None)
    epcScore = st.selectbox('EPC score', get_values_from_df('epcScore', df), index=None)
    terraceOrientation = st.selectbox('Terrace orientation', get_values_from_df('terraceOrientation', df), index=None)
    gardenOrientation = st.selectbox('Garden orientation', get_values_from_df('gardenOrientation', df), index=None)
    floodZoneType = st.selectbox('Flood zone type', get_values_from_df('floodZoneType', df), index = 2, format_func = lambda x: x.replace('_', ' ') if isinstance(x, str) else '--')

    bedroomCount = st.number_input('Bedroom count', min_value = 0, max_value = 10, value = 2)
    kitchenType = st.selectbox('Kitchen type', get_values_from_df('kitchenType', df), index=None)
    toiletCount = st.number_input('Toilet count', min_value = 1, max_value = 10, value = 1)
    bathroomCount = st.number_input('Bathroom count', min_value = 1, max_value = 10, value = 1)
    habitableSurface = st.number_input('Habitable surface', min_value = 10, max_value = 500, value = 50)
    kitchenSurface = st.number_input('Kitchen surface', min_value = 5, max_value = 100, value = 10)
    facedeCount = st.number_input('Facade count', min_value = 1, max_value = 8, value = 1)
    streetFacadeWidth = st.number_input('Street facade width', min_value = 1, max_value = 20, value = 5)
    terraceSurface = st.number_input('Terrace surface', min_value = 0, max_value = 30, value = 0)
    gardenSurface = st.number_input('Garden surface', min_value = 0, max_value = 1000, value = 0)
    landSurface = st.number_input('Land surface', min_value = 0, max_value = 10000, value = 0)

    prediction = ''
    col1, col2, col3 = st.columns(3)
    with col2 :
        if col2.button('Predict') :
            data = {'bedroomCount': bedroomCount,
                'habitableSurface': habitableSurface,
                'facedeCount': facedeCount,
                'streetFacadeWidth': streetFacadeWidth,
                'kitchenSurface': kitchenSurface,
                'landSurface': landSurface,
                'terraceSurface': terraceSurface,
                'gardenSurface': gardenSurface,
                'toiletCount': toiletCount,
                'bathroomCount': bathroomCount,
                'type': tp,
                'subtype': subtype,
                'postCode': int(postCode),
                'hasBasement': 1 if st.session_state['hasBasement'] else 0,
                'buildingCondition': buildingCondition,
                'buildingConstructionYear': buildingConstructionYear,
                'hasTerrace': 1 if st.session_state['hasTerrace'] else 0,
                'floodZoneType': floodZoneType,
                'heatingType': heatingType,
                'kitchenType': kitchenType,
                'gardenOrientation': gardenOrientation,
                'hasSwimmingPool': 1 if st.session_state['hasSwimmingPool'] else 0,
                'terraceOrientation': terraceOrientation,
                'epcScore': epcScore}
            
            for key in data.keys() :
                if data[key] == None :
                    data[key] = ''
            
            prediction = requests.post(url=API_ENDPOINT, json=data).json()

    st.success(prediction if isinstance(prediction, str) or ('price' not in prediction) else "Predicted price is %.2f€" % prediction['price'])

if __name__ == "__main__":
    main()