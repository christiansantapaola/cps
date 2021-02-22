# import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
# import seaborn as sns
import os

# https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf


def reshape(df):
    """reshape the dataframe in a more convenient form"""
    # melt transform column to row
    # in particular transform the columns of the years ('1800', '1801', ...., '2020')
    # into a single column Year and the value of the row get moved into a column "Value"
    # each pair(year, value) is identified by the id_vars columns, which remains unchanged.
    df = pd.melt(df,
                 id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code', 'Attribute'],
                 value_vars=[str(i) for i in range(1800, 2021)], var_name='Years', value_name="Value")
    # pivot: transform the row of column into columns themself.
    # the rows of the columns "Indicator Name" become columns.
    # the rows of the columns "Value" become the value of the new columns
    # the columns ["Country Name", "Years"] become the new index of the dataframe
    # each pair (country, years) identify uniquely a row of the new dataframe
    # the columns not listed here get dropped.
    df = df.pivot(index=['Country Name', 'Years'], columns='Indicator Name', values='Value')
    # reset_index():
    # the index constructed by the pivot operation is multi-structured [country, ['1800',...,'2020']]
    # we want to flatten it ([country, '1800'], ..., [country, '2020'])
    # so we use reset index to create a new index (0...N) and flatten the multi structured index using
    # the `inplace` option.
    df.reset_index(inplace=True)
    return df


def get_indicator(df):
    """get all the indicator of the dataframe"""
    indicator = list(df.columns)
    return indicator


def get_all_country(df):
    """return the name of all the country"""
    country = df["Country Name"].drop_duplicates()
    return country


def get_all_country_with_at_least_a_missing_value(df):
    """return the name of all the country that have at least one missing indicator"""
    country_with_missing_data = df[(df['Debt to GDP Ratio'] != df['Debt to GDP Ratio'])
                                   |
                                   (df['Gross Domestic Product'] != df['Gross Domestic Product'])
                                   |
                                   (df['Gross Government Debt'] != df['Gross Government Debt'])]
    return country_with_missing_data["Country Name"].drop_duplicates()


def plot_index(df, nations, index, output_file):
    """
    df: pandas dataframe
    nations: list of nation, es: ["Italy", "France", "Germany"]
    index: list of 2 elements, Index and abbrevation without space. es: ["Gross Government Debt", "GGD"]
    plot the choosen index for a group of nations.
    """
    # i take the pair (year, gdp) for the chosen nations
    nation_df = pd.DataFrame()
    for nation in nations:
        nation_data = df[['Country Name', 'Years', index[0]]].rename(columns={
            'Country Name': 'Country',
            index[0]: index[1]
            }).query('(Country == "{}") & ({} ==  {})'.format(nation, index[1], index[1]))
        nation_df = pd.concat([nation_df, nation_data])
    # concatenating the data collected
    # pivot the data so i can plot multiple lines.
    plotting_data = nation_df.pivot(index="Years", columns="Country", values=["{}".format(index[1])])
    # set up the plot.
    plotting_data.plot.line()
    plt.ylabel(index[1])
    plt.xlabel('Year')
    plt.xticks(rotation=80)
    plt.savefig(output_file)


def get_percentile_index_ratio_growth(df, nation, index):
    """
    :param df: dataframe
    :param nation: nation to calculate
    :param index: [index, index_shorthand]
    :return: df
    get percentile growth of index = [index, shorthand] for specifying nation.
    the df must be ordered by year.
    """

    nation_data = df[['Country Name', 'Years', index[0]]].rename(columns={
        'Country Name': 'Country',
        index[0]: index[1]
    }).query('(Country == "{}") & ({} ==  {})'.format(nation, index[1], index[1]))
    # method to calculate percentile growth
    df['{} growth %'.format(index[1])] = nation_data['{}'.format(index[1])].pct_change()
    return df

def label_point(x, y, val, ax):
    """utility function for scatter plotting"""
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x'], point['y'], str(point['val']))

def scatter_plot(df, nations, year, output_file, plot_name=False):
    """
    df: pandas dataframe
    nations: list of nation, es: ["Italy", "France", "Germany"]
    index: list of 2 elements, Index and abbrevation without space. es: ["Gross Government Debt", "GGD"]
    plot the choosen index for a group of nations.
    """
    # i take the pair (year, gdp) for the chosen nations
    nation_df = pd.DataFrame()
    for nation in nations:
        nation_data = df[['Country Name', 'Years', 'Gross Domestic Product', 'Gross Government Debt']].rename(columns={
            'Country Name': 'Country',
            'Gross Domestic Product': 'GDP',
            'Gross Government Debt': 'GGD'
            }).query('(Country == "{}") & (Years == "{}") &({} ==  {}) & ({} == {})'.format(nation, year, 'GDP',
                                                                                             'GDP', 'GGD', 'GGD'))
        nation_df = pd.concat([nation_df, nation_data])
    # concatenating the data collected
    # pivot the data so i can plot multiple lines.
    plotting_data = nation_df
    # set up the plot.
    pd.set_option('display.max_columns', None)
    print(df[(df['Country Name'] == "Hungary") & (df['Years'] == year)])
    print(plotting_data)
    ax = plotting_data.plot.scatter(x='GDP', y='GGD', alpha=0.5, title=year)
    if plot_name:
        for ind, country in enumerate(plotting_data['Country']):
            ax.annotate(country, (plotting_data['GDP'].iloc[ind], plotting_data['GGD'].iloc[ind]))
    # force matplotlib to draw the graph
    plt.savefig(output_file)
    



def main():
    DATA_FOLDER = os.path.abspath('data')
    OUTPUT_FOLDER = os.path.abspath('output')
    DATA_FILE = os.path.join(DATA_FOLDER, 'data.csv')
    df = pd.read_csv(DATA_FILE)
    df = reshape(df)
    print(df)
    indicator = get_indicator(df)
    print(indicator)
    print("---------------------------------------------------------------------")
    country = get_all_country(df)
    print(country)
    print("---------------------------------------------------------------------")
    missing_data_country = get_all_country_with_at_least_a_missing_value(df)
    print(missing_data_country)
    print("---------------------------------------------------------------------")
    plot_index(df, ["Italy", "Germany", "France"], ["Gross Domestic Product", "GDP"], os.path.join(OUTPUT_FOLDER, 'IGF_GDP.png'))
    plot_index(df, ["Italy", "Germany", "France"], ["Gross Government Debt", "GGD"], os.path.join(OUTPUT_FOLDER, 'IGF_GGD.png'))
    for nation in country:
        get_percentile_index_ratio_growth(df, nation, ["Gross Domestic Product", "GDP"])
        get_percentile_index_ratio_growth(df, nation, ["Gross Government Debt", "GDD"])
    #eu_nation = ['Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czechia', 'Denmark',
    #    'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary',
    #    'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg', 'Malta', 'Netherlands',
    #    'Poland', 'Portugal', 'Romania', 'Slovakia', 'Slovenia', 'Spain', 'Sweden']
    eu_nation = ['Austria', 'Belgium', 'Denmark', 'France', 'Germany', 'Greece', 'Italy', 'Netherlands', 'Sweden', 'Spain']
    scatter_plot(df, eu_nation, '2001', os.path.join(OUTPUT_FOLDER, '2001_EU_SCATTER_NO_NAME.png'))
    scatter_plot(df, eu_nation, '2005', os.path.join(OUTPUT_FOLDER, '2005_EU_SCATTER_NO_NAME.png'))
    scatter_plot(df, eu_nation, '2001', os.path.join(OUTPUT_FOLDER, '2001_EU_SCATTER_WITH_NAME.png'), plot_name=True)
    scatter_plot(df, eu_nation, '2005', os.path.join(OUTPUT_FOLDER, '2005_EU_SCATTER_WITH_NAME.png'), plot_name=True)

if __name__ == "__main__":
    main()
