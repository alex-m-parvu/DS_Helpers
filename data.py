import pandas as pd
from meteostat import Hourly, Point
import pvlib
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import silhouette_score
import holidays
from sklearn.base import BaseEstimator, TransformerMixin



def other_generator(series: pd.Series, other_threshold=5):
    series = pd.Series(series)

    value_counts = series.value_counts()

    filter = value_counts[ value_counts>other_threshold ]

    return series.apply( lambda x:x if x in filter.index else "other" )


def other_generator_columns(df:pd.DataFrame, other_threshold=20):

    new_df = df.copy()

    object_column = new_df.select_dtypes(include='object').columns

    for col in object_column:
        new_df[col] = other_generator(series=new_df[col], other_threshold=other_threshold)

    return new_df



def get_days_to_nearest_holiday(date_input, country_code='RO'):
    """
    Calculate the number of days to the nearest public or religious holiday in a given country.

    This function takes a date input and an optional country code (default is Romania),
    and returns the number of days to the nearest holiday in that country.
    The holidays are determined using the holidays library.

    Parameters:
        date_input (datetime, pd.Timestamp, or pd.DatetimeIndex): The date for which to calculate the distance to the nearest holiday.
        country_code (str): A two-letter code specifying the country. See `holidays` library documentation for available codes. Defaults to 'RO' (Romania).

    Returns:
        int: The number of days to the nearest holiday.

    Example:
        >>> from datetime import datetime
        >>> get_days_to_nearest_holiday(datetime(2024, 1, 1))
        0

        >>> import pandas as pd
        >>> get_days_to_nearest_holiday(pd.Timestamp('2024-01-02'))
        1

    Note:
        - The function assumes the input date is in the Gregorian calendar.
        - The holidays are specific to the given country and are fetched using the `holidays` library.
    """
    # Convert input to datetime if needed
    if isinstance(date_input, pd.DatetimeIndex):
        date_input = date_input[0]
    elif isinstance(date_input, pd.Timestamp):
        date_input = date_input.to_pydatetime()

    # Initialize holidays for the given country
    try:
        country_holidays = holidays.CountryHoliday(country_code)
    except KeyError:
        raise ValueError(f"Unsupported country code: {country_code}")

    # Get holidays within a reasonable range around input date
    year = date_input.year
    holiday_dates = [
        dt for dt in country_holidays[f'{year-1}-12-01':f'{year+1}-01-31']
    ]

    # Calculate distances to all holidays
    distances = [
        abs((date_input.date() - holiday).days)
        for holiday in holiday_dates
    ]

    # Return minimum distance
    return min(distances) if holiday_dates else None



def find_optimal_clusters(df, algorithm=KMeans(), max_clusters=10):
    """
    Determine the optimal number of clusters using the Silhouette Score.

    Parameters:
        df (pd.DataFrame): Input dataset.
        algorithm (sklearn.cluster.BaseEstimator): Clustering algorithm to use. Defaults to KMeans().
        max_clusters (int, optional): Maximum number of clusters to consider. Defaults to 10.

    Returns:
        int: Optimal number of clusters for the given dataset.
    """

    # Initialize the optimal score and the optimal number of clusters
    best_score = -1
    optimal_n_clusters = None

    # Try out all possible numbers of clusters from 2 to max_clusters
    for n_clusters in range(2, max_clusters + 1):
        algorithm.n_clusters = n_clusters
        labels = algorithm.fit_predict(df)
        score = silhouette_score(df, labels)

        # Update the optimal score and the optimal number of clusters if necessary
        if score > best_score:
            best_score = score
            optimal_n_clusters = n_clusters

    return optimal_n_clusters

class EmbeddingCreator_sk(BaseEstimator, TransformerMixin):
    """
    Scikit-learn-compatible transformer for embedding categorical IDs.

    A scikit-learn-compatible transformer that converts categorical IDs into dense embedding vectors.

    This transformer learns an embedding matrix during `fit` and maps each category (ID) to a fixed-size
    vector. It can optionally use random or sequential embeddings and handles unknown categories during
    `transform` by assigning them to an "other" embedding.

    Parameters
    ----------
    embedding_dims : int, default=8
        The dimensionality of the embedding vectors.

    random : bool, default=False
        If True, the embeddings are initialized randomly using a standard normal distribution.
        If False, embeddings are initialized using a sequence of increasing numbers.

    prefix : str or None, default=None
        Optional prefix for the embedding column names in the output DataFrame.
        If None, columns will be named 'V_0', 'V_1', ..., otherwise '{prefix}V_0', '{prefix}V_1', ...

    Attributes
    ----------
    id_map : dict
        A mapping from category ID to row index in the embedding matrix. Includes a special 'other' key
        for unknown categories.

    matrix : ndarray of shape (n_categories + 1, embedding_dims)
        The embedding matrix. Each row is a vector representation for a category, with the last row
        reserved for the 'other' category.

    Examples
    --------
    >>> from sklearn.pipeline import Pipeline
    >>> emb = EmbeddingCreator(embedding_dims=4, random=True, prefix='user_')
    >>> X = ['user1', 'user2', 'user3']
    >>> emb.fit_transform(X)
         user_V_0  user_V_1  user_V_2  user_V_3
    0   ...       ...       ...       ...
    1   ...       ...       ...       ...
    2   ...       ...       ...       ...
    """
    def __init__(self, embedding_dims=8, random=False, prefix=None):
        self.embedding_dims = embedding_dims
        self.random = random
        self.prefix = prefix
        self.id_map = {}
        self.matrix = None

    def fit(self, X, y=None):
        unique_ids = sorted(set(X))  # assumes X is 1D array-like
        self.id_map = {id_: i for i, id_ in enumerate(unique_ids)}
        self.id_map['other'] = len(self.id_map)

        n_ids = len(self.id_map)
        if self.random:
            self.matrix = np.random.normal(size=(n_ids, self.embedding_dims))
        else:
            self.matrix = np.arange(0, n_ids * self.embedding_dims).reshape(n_ids, self.embedding_dims)

        return self

    def transform(self, X):
        if self.matrix is None:
            raise ValueError("Call fit before transform.")

        ids = [x if x in self.id_map else 'other' for x in X]
        indices = [self.id_map[id_] for id_ in ids]

        columns = [f'{self.prefix}V_{i}' if self.prefix else f'V_{i}' for i in range(self.embedding_dims)]
        return pd.DataFrame(self.matrix[indices], columns=columns)


class EmbeddingCreator:
    """
    A class for creating embeddings for a list of IDs.

    Attributes:
        embedding_dims (int): Number of dimensions for the embedding vectors. Default is 8.
        random (bool): If True, creates random embeddings using normal distribution. If False, creates sequential number embeddings. Default is False.
        prefix (str): Prefix to add to column names. If None, columns will be named 'V_0', 'V_1', etc. If provided, columns will be named '{prefix}V_0', '{prefix}V_1', etc. Default is None.
        id_map (dict): A dictionary mapping IDs to their corresponding indices in the embedding matrix.
        matrix (numpy.ndarray): The embedding matrix.
    """

    def __init__(self, embedding_dims=8, random=False, prefix=None):
        self.embedding_dims = embedding_dims
        self.random = random
        self.prefix = prefix
        self.id_map = {}
        self.matrix = None

    def fit(self, ids):
        """
        Fit the embedding creator to a list of IDs.

        Parameters:
            ids (list-like): List of identifiers to create embeddings for. Can contain duplicates.
        """
        unique_ids = sorted(list(set(ids)))  # sorted for consistency
        self.id_map = {id_: i for i, id_ in enumerate(unique_ids)}
        self.id_map['other'] = max(self.id_map.values())+1  # add an 'other' category for new IDs not encountered during fit

        if self.random:
            self.matrix = np.random.normal(size=(len(unique_ids)+1, self.embedding_dims))  # add an extra row for the 'other' category
        else:
            self.matrix = np.arange(0, (len(unique_ids)+1) * self.embedding_dims).reshape(len(unique_ids)+1, self.embedding_dims)  # add an extra row for the 'other' category

    def create_embedding(self, ids):
        """
        Create an embedding matrix for a list of IDs.

        Parameters:
            ids (list-like): List of identifiers to create embeddings for. Can contain duplicates.

        Returns:
            pandas.DataFrame: DataFrame with shape (len(ids), embedding_dims) containing the embeddings.
                              Each row corresponds to the embedding for the ID at the same position in the input list.
        """
        if self.matrix is None:
            raise ValueError("Fit method must be called before creating embeddings")

        ids = list(map(lambda x: x if x in self.id_map.keys() else 'other', ids))  # in case new ids that were not encountered during the fit method are used

        indices = [self.id_map[id_] for id_ in ids]
        columns = [f'{self.prefix}V_{i}' for i in range(self.embedding_dims)]
        result = pd.DataFrame(self.matrix[indices], columns=columns)

        return result

    def transform(self, ids):
        """
        Create embeddings for a list of IDs.

        Parameters:
            ids (list-like): List of identifiers to create embeddings for. Can contain duplicates.

        Returns:
            pandas.DataFrame: DataFrame with shape (len(ids), embedding_dims) containing the embeddings.
                              Each row corresponds to the embedding for the ID at the same position in the input list.
        """
        return self.create_embedding(ids)
    
    def fit_transform(self, ids):
        """
        Fit the embedding creator to the given ids and transform them into embeddings.

        Parameters:
            ids (list-like): List of identifiers to create embeddings for. Can contain duplicates.

        Returns:
            pandas.DataFrame: DataFrame with shape (len(ids), embedding_dims) containing the embeddings.
                              Each row corresponds to the embedding for the ID at the same position in the input list.

        Note:
            This method should only be called on the training set, as training the embedding object on the test set can lead to overfitting and biased results.
        """
        self.fit(ids)  # fit the embedding creator to the given ids
        return self.create_embedding(ids)  # transform the ids into embeddings



def find_embed_object_cols(df:pd.DataFrame, embedding_dims=4):

    df = df.copy()

    object_columns = df.select_dtypes(include='object').columns

    shape_before = df.shape


    embedings ={}

    for col in object_columns:

    # col = object_columns[0]

        print(col)
        embeding = EmbeddingCreator(embedding_dims=embedding_dims, prefix=col+'_')
        embeding_df = embeding.fit_transform(ids=df[col])
        embeding_df.index = df.index
        df = pd.concat(objs=(df, embeding_df), axis=1)

        df = df.drop(labels=col, axis=1)


        embedings[col] = embeding

    shape_after = df.shape

    result = dict(shape_before=shape_before, shape_after=shape_after, embedings=embedings, df=df  )

    return result



# Create a function for creating embedings 

def create_embedding(ids
                     , embedding_dims=8
                     , random=False
                     , prefix=None):
    """Create an embedding matrix for a list of IDs.

    Parameters:
    -----------
    ids : list-like
        List of identifiers to create embeddings for. Can contain duplicates.
    embedding_dims : int, default=8
        Number of dimensions for the embedding vectors.
    random : bool, default=False
        If True, creates random embeddings using normal distribution.
        If False, creates sequential number embeddings.
    prefix : str, optional
        Prefix to add to column names. If None, columns will be named 'V_0', 'V_1', etc.
        If provided, columns will be named '{prefix}V_0', '{prefix}V_1', etc.

    Returns:
    --------
    pandas.DataFrame
        DataFrame with shape (len(ids), embedding_dims) containing the embeddings.
        Each row corresponds to the embedding for the ID at the same position in the input list.

    Examples:
    --------
    >>> ids = [1, 2, 1, 3]
    >>> create_embedding(ids, embedding_dims=3, random=True, prefix='emb_')
       emb_V_0  emb_V_1  emb_V_2
    0    0.123    0.456    0.789
    1    0.234    0.567    0.890
    2    0.123    0.456    0.789
    3    0.345    0.678    0.901
    """
    # Get unique ids and create a mapping
    unique_ids = sorted(list(set(ids)))  # sorted for consistency
    id_map = {id_: i for i, id_ in enumerate(unique_ids)}
    
    # Create the base matrix
    if random:
        matrix = np.random.normal(size=(len(unique_ids), embedding_dims))
    else:
        matrix = np.arange(0, len(unique_ids) * embedding_dims).reshape(len(unique_ids), embedding_dims)
    
    # Create column names
    if prefix is not None:
        columns = [f'{prefix}V_{i}' for i in range(embedding_dims)]
    else:
        columns = [f'V_{i}' for i in range(embedding_dims)]
    
    # Map input ids to matrix indices
    indices = [id_map[id_] for id_ in ids]
    
    # Create DataFrame with proper indexing
    result = pd.DataFrame(matrix[indices], columns=columns)
    
    return result

def get_wheater_hourly(start
                       , end
                       , latitude
                       , longitude
                       , alt = None
                       , tz=None):
    
    """
    Retrieves hourly weather data for a specific geographical location and time period.

    Parameters:
    ----------
    start : datetime
    Start date and time for the weather data request
    
    end : datetime
    End date and time for the weather data request
    
    latitude : float
    Latitude of the location
    
    longitude : float
    Longitude of the location
    
    alt : float, optional
    Altitude of the location in meters. If None, altitude is not considered
    
    tz : str, optional
    Timezone for the timestamp index. If None, timestamps will not be localized

    Returns:
    -------
    pandas.DataFrame
    DataFrame containing hourly weather data for the specified location and time period
    - index: timestamp - Hourly timestamps
    - columns: Various weather parameters (temperature, humidity, etc.)

    Notes:
    -----
    Uses the Point and Hourly classes (likely from a weather API service) to fetch the data.
    Timezone localization is applied to the result if tz parameter is provided.
    """
    
    point = Point(lat = latitude, lon=longitude, alt=alt)
    weather = Hourly(point, start.tz_localize(None), end.tz_localize(None))
    weather = weather.fetch()
    if tz is not None:
        weather.index = weather.index.tz_localize(tz, ambiguous='NaT')
        
    return weather


def GHI(start, end, latitude, longitude, tz='UTC'):
    
    """
    Calculates the Global Horizontal Irradiance (GHI) for a specific location and time period.

    Parameters:
    ----------
    start : datetime
    Start date and time for the GHI calculation
    
    end : datetime
    End date and time for the GHI calculation
    
    latitude : float
    Latitude of the location
    
    longitude : float
    Longitude of the location
    
    tz : str, default='UTC'
    Timezone for the calculation and output

    Returns:
    -------
    pandas.Series
    Hourly Global Horizontal Irradiance values
    - index: timestamp - Hourly timestamps in specified timezone
    - values: GHI in kWh/m²

    Notes:
    -----
    - Uses pvlib's simplified Solis model for clear-sky irradiance calculations
    - Converts the original GHI values from W/m² to kWh/m²
    - Returns hourly energy values suitable for solar production estimates
    """
    
    # Create a range of times
    times = pd.date_range(start=start, end=end, freq='h', tz=tz)
    
    # Create a Location object
    location = pvlib.location.Location(latitude, longitude)
    
    # Calculate clear-sky irradiance using
    clearsky_irradiance = location.get_clearsky(times, model='simplified_solis')
    
    # Extract Global Horizontal Irradiance (GHI) for each hour
    ghi = clearsky_irradiance['ghi']
    
    # Convert GHI from W/m^2 to kWh/m^2 
    hourly_energy = ghi * 1 / 1000  # kWh/m^2
    
    return hourly_energy #kWh/m²


def GHIC(start, end, latitude, longitude, cloud_cover=None, tz='UTC'):
    """
    Calculate the actual solar energy reaching the ground considering cloud cover.

    Parameters:
    - start: Start date and time (string or datetime)
    - end: End date and time (string or datetime)
    - latitude: Latitude of the location
    - longitude: Longitude of the location
    - cloud_cover: A pandas Series or list of cloud cover values (0 to 9) for each hour
    - tz: Timezone (default is 'UTC')

    Returns:
    - A pandas Series with the actual solar energy in kWh/m² for each hour
    """
    
    if cloud_cover is None:
        Bucharest = Point(lat = latitude, lon=longitude)
        weather = Hourly(Bucharest, start.tz_localize(None), end.tz_localize(None))
        weather = weather.fetch()
        
        if not weather.index.is_unique:# in case there are duplicated values
            weather = weather.groupby(weather.index).mean()
        
        
        weather.index = weather.index.tz_localize(tz
                                                  , ambiguous='NaT' # in case there are any issues with missing values
                                                  , nonexistent='shift_forward') # in case of daylight savings
        cloud_cover = weather.coco
        
        cloud_cover = pd.Series(cloud_cover)
    
    # Create a range of times
    times = pd.date_range(start=start
                          , end=end, freq='h'
                          , tz=tz
                          , nonexistent='shift_forward')# in case of daylight savings time
    
    # Create a Location object
    location = pvlib.location.Location(latitude, longitude)
    
    # Calculate clear-sky irradiance using the 'simplified_solis' model
    clearsky_irradiance = location.get_clearsky(times, model='simplified_solis')
    
    # Extract Global Horizontal Irradiance (GHI) for each hour
    ghi_clear_sky = clearsky_irradiance['ghi']
    
    # Convert cloud cover to a fraction (0 to 1)
    cloud_cover_fraction = cloud_cover/30
    
    # Calculate the actual GHI considering cloud cover
    ghi_actual = ghi_clear_sky * (1 - cloud_cover_fraction)
    
    # Convert GHI from W/m^2 to kWh/m^2
    actual_energy = ghi_actual * 1 / 1000  # kWh/m^2
    actual_energy.name = 'ghic'
    actual_energy = actual_energy.groupby(level=0).mean() # in case of duplicated indexes
    return actual_energy  # kWh/m²
