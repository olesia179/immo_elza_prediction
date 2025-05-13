import pandas as pd
import numpy as np

class Cleaner :

    __dataset_path = "./data/Kangaroo.csv"

    @staticmethod
    def load_data(self, path: str) -> pd.DataFrame :
        '''
            Load the dataset
            :return: DataFrame
        '''
        return pd.read_csv(path)
    
    @staticmethod
    def drop_duplicates(data: pd.DataFrame) -> pd.DataFrame :
        '''
            Remove duplicates from the dataset
            :param data: DataFrame to clean
            :return: cleaned DataFrame
            Rows:
                Indexes 5591 - 440319 is duplicate id: 20663057.0
        '''
        data.drop_duplicates(subset=['id'], keep='last', inplace=True)
        return data
    
    @staticmethod
    def drop_na(data: pd.DataFrame) -> pd.DataFrame :
        '''
            Remove rows with NaN values in price column
            :param data: DataFrame to clean
            :return: cleaned DataFrame
        '''
        data.dropna(subset=['price'], how='all', inplace=True)
        return data
    
    @staticmethod
    def drop_columns(data: pd.DataFrame) -> pd.DataFrame :
        '''
            Remove columns from the dataset
            :param data: DataFrame to clean
            :return: cleaned DataFrame
            Columns:
                0 index
                1 id
                2 url
                # 4 subtype
                11 roomCount
                12 monthlyCost
                34 hasBalcony
                35 hasGarden
                # 38 parkingCountIndoor
                # 39 parkingCountOutdoor
                50 accessibleDisabledPeople
        '''
        data.drop(data.columns[[0, 1, 2, 11, 12, 34, 35, 50]], axis=1, inplace=True)
        return data

    @staticmethod
    def clean_epcScore(data: pd.DataFrame) -> pd.DataFrame :
        '''
            Replace strange values in epcScore column with NaN
            :param data: DataFrame to clean
            :return: cleaned DataFrame
        '''
        values_to_replace = ['C_A', 'F_C', 'G_C', 'D_C', 'F_D', 'E_C', 'G_E', 'E_D', 'C_B', 'X', 'G_F']
        data.loc[data['epcScore'].isin(values_to_replace), 'epcScore'] = np.nan
        return data
    
    @staticmethod
    def replace_outlier_toiletCount(data: pd.DataFrame) -> pd.DataFrame :
        '''
            Replace outlier value in toiletCount column with 2
            :param data: DataFrame to clean
            :return: cleaned DataFrame
        '''
        data.loc[data['toiletCount'] == 1958, 'toiletCount'] = 2
        return data 
    
    @staticmethod
    def float_to_int(data: pd.DataFrame) -> pd.DataFrame :
        '''
            Convert columns to integer
            :param data: DataFrame to clean
            :return: cleaned DataFrame
        '''
        to_int = lambda x : int(x)
        # 'id', 'roomCount'
        cols_to_int = ['bedroomCount', 'bathroomCount',
                        'habitableSurface', 'diningRoomSurface', 'kitchenSurface',
                        'landSurface', 'livingRoomSurface', 'gardenSurface',
                        'terraceSurface', 'buildingConstructionYear', 'facedeCount',
                        'floorCount', 'toiletCount']
        for col in cols_to_int :
            data[col] = data[col].fillna(data[col].median()).apply(to_int)
        return data
    
    @staticmethod
    def bool_to_int(data: pd.DataFrame) -> pd.DataFrame :
        '''
            Convert boolean columns to integer
            :param data: DataFrame to clean
            :return: cleaned DataFrame
        '''
        to_bool = lambda x : False if np.isnan(x) else x
        bool_to_int = lambda x : 1 if x == True else 0
        # 'hasGarden'
        cols_to_bool = ['hasAttic', 'hasBasement', 'hasDressingRoom',
                        'hasDiningRoom', 'hasLift', 'hasHeatPump',
                        'hasPhotovoltaicPanels', 'hasThermicPanels',
                        'hasLivingRoom', 'hasAirConditioning', 'hasArmoredDoor',
                        'hasVisiophone', 'hasOffice', 'hasSwimmingPool',
                        'hasFireplace', 'hasTerrace']
        for col in cols_to_bool :
            # mode_val = data[col].isnull().count() .apply(to_bool).mode()[0]
            data[col] = data[col].apply(bool_to_int)
        return data
    
    @staticmethod
    def round_float(data: pd.DataFrame) -> pd.DataFrame :
        '''
            Round float columns to 2 decimal places
            :param data: DataFrame to clean
            :return: cleaned DataFrame
        '''
        round_float = lambda x : round(float(x), 2)
        
        cols_to_round = ['streetFacadeWidth']
        for col in cols_to_round :
            data[col] = data[col].fillna(data[col].median()).apply(round_float)
        return data
    
    @staticmethod
    def type_to_int(data: pd.DataFrame) -> pd.DataFrame :
        '''
            Replace type column with 0 if APARTMENT, 1 if HOUSE
            :param data: DataFrame to clean
            :return: cleaned DataFrame
        '''
        type_to_int = lambda x : 0 if x == 'APARTMENT' else 1

        data['type'] = data['type'].apply(type_to_int)
        return data

    @staticmethod
    def locality_to_upper(data: pd.DataFrame) -> pd.DataFrame :
        '''
            Convert locality column to uppercase
            :param data: DataFrame to clean
            :return: cleaned DataFrame
        '''
        data['locality'] = data['locality'].str.upper()
        return data
    
    @staticmethod
    def get_rid_of_outliers_col(data: pd.DataFrame, column_name: str) -> pd.DataFrame :
        '''
            Remove outliers from the dataset
            :param data: DataFrame to clean
            :return: cleaned DataFrame
        '''
        # Remove outliers from the column
        q1 = data[column_name].quantile(0.25)
        q3 = data[column_name].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        data = data[(data[column_name] >= lower_bound) & (data[column_name] <= upper_bound)]
        return data
    
    @staticmethod
    def clean_data() -> pd.DataFrame :
        return (
                pd.DataFrame().pipe(Cleaner.load_data, Cleaner.__dataset_path)
                .pipe(Cleaner.drop_duplicates)
                .pipe(Cleaner.drop_na)
                .pipe(Cleaner.get_rid_of_outliers_col, 'price')
                .pipe(Cleaner.get_rid_of_outliers_col, 'habitableSurface')
                .pipe(Cleaner.drop_columns)
                .pipe(Cleaner.clean_epcScore)
                .pipe(Cleaner.replace_outlier_toiletCount)
                .pipe(Cleaner.float_to_int)
                .pipe(Cleaner.bool_to_int)
                .pipe(Cleaner.round_float)
                .pipe(Cleaner.locality_to_upper)
            )