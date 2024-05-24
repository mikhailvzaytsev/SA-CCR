import pandas as pd
import numpy as np
import math
from scipy.stats import norm
import cmath

def RC(deals):
    dealsMTM = deals.groupby(['counterparty', 'csa', 'TH', 'MTA']).apply(
        lambda x: pd.Series({
            'mtm': (x['mtm'] * x['rate']).sum(),
            'collateral': (x['collateral'] * x['rate']).sum(),
            'NICA': x['NICA'].sum()
        })).reset_index()

    dealsMTM['RC'] = np.where((dealsMTM['csa'] == 0), np.maximum(dealsMTM['mtm'] - dealsMTM['collateral'], 0),
                  np.where((dealsMTM['csa'] == 1), np.maximum(dealsMTM['mtm'] - dealsMTM['collateral'], np.maximum(dealsMTM['TH'] + dealsMTM['MTA'] - dealsMTM['NICA'], 0)), np.nan))

    deals = deals.merge(dealsMTM[['counterparty', 'RC']], how='left', on='counterparty')

    return deals

def sduration_calc(deals):
    deals['sduration'] = np.where(deals['product'].isin(['IRS','ESO','USO','CD_ААА','CD_AA','CD_A',
                                                                     'CD_BBB','CD_BB','CD_B','CD_CCC','CI_IG','CI_SG']), (np.exp(-0.05 * deals['year_frac_s']) - np.exp(-0.05 * deals['year_frac_e'])) / 0.05,
                         np.where(~(deals['product'].isin(['IRS', 'ESO', 'USO', 'CD_ААА', 'CD_AA', 'CD_A',
                                                                     'CD_BBB','CD_BB','CD_B','CD_CCC','CI_IG','CI_SG'])), 1, np.nan))
    return deals

def adj_notional_calc(deals):
    deals['adj_notional'] = deals['sduration'] * deals['notional_1']/1000 * deals['rate']
    return deals

def sdelta_calc(deals):

    #to divide df into 2 dfs to for supervisory delta calculation on non-linear instruments
    deals_options = deals[deals['product'].isin(['FXO', 'ESO', 'ASO'])].copy()
    deals_linear = deals[~(deals['product'].isin(['FXO', 'ESO', 'ASO']))].copy()

    deals_options['koef'] = np.where((deals_options['option_side'] == 'call') & (deals_options['asset_liab'] == 1), 1,
                            np.where((deals_options['option_side'] == 'call') & (deals_options['asset_liab'] == -1),-1,
                            np.where((deals_options['option_side'] == 'put') & (deals_options['asset_liab'] == 1), -1,
                            np.where((deals_options['option_side'] == 'put') & (deals_options['asset_liab'] == -1), 1, np.nan))))

    deals_options['sd_sign'] = np.where((deals_options['option_side'] == 'call') & (deals_options['asset_liab'] == 1), 1,
                               np.where((deals_options['option_side'] == 'call') & (deals_options['asset_liab'] == -1),1,
                               np.where((deals_options['option_side'] == 'put') & (deals_options['asset_liab'] == 1), -1,
                               np.where((deals_options['option_side'] == 'put') & (deals_options['asset_liab'] == -1), -1, np.nan))))

    deals_options['sdelta'] = deals_options['koef'] * norm.cdf((deals_options['sd_sign'] * (math.log(0.06 / 0.05) + 1 / 2 * deals_options['Supervisory_option_volatility'] ** 2 * deals_options['year_frac_s']) / (deals_options['Supervisory_option_volatility'] * deals_options['year_frac_s'] ** (1 / 2))), loc=0, scale=1)
    deals_options = deals_options.drop(['koef', 'sd_sign'], axis=1)

    deals_linear['sdelta'] = deals_linear['asset_liab']

    deals_all = pd.concat([deals_linear, deals_options])

    return deals_all

def maturity_factor_calc(deals):
    deals['maturity_factor'] = (np.minimum(np.maximum(10 / 250, deals['year_frac_m']), 1)) ** (1 / 2)
    deals.loc[(deals['csa'] == 1), 'maturity_factor'] = 1.5 * (14 / 250) ** (1 / 2)

    return deals

def maturity_bucket(deals):
    deals['mat_bucket'] = np.where((deals['year_frac_m'] < 1), 1,
                          np.where(((1 <= deals['year_frac_m']) & (deals['year_frac_m'] < 5)), 2,
                          np.where((deals['year_frac_m'] >= 5), 3, np.nan)))

    return deals

def hedging_sets_calc(deals):
    #grouping instruments into maturity buckets for each counterparty
    df = deals.groupby(['counterparty', 'product', 'ccy_1', 'rate', 'mat_bucket', 'Supervisory_factor', 'RC']).apply(
        lambda x:
        pd.Series({
            'D': (x['sdelta'] * x['adj_notional'] * x['maturity_factor']).sum(),
            'uncoll_mtm': (x['mtm'] - x['collateral']).sum()
        })).reset_index()
    df['hedging_set'] = df['product'] + '_' + df['ccy_1']

    unique_hs = df['hedging_set'].unique()

    data = pd.DataFrame(columns=['counterparty', 'product', 'rate', 'Supervisory_factor', 'hedging_set', 'uncoll_mtm', 'RC', 'effective_notional'])
    #to calculate effective notional in every time bucket for each hedging set separately
    for hs_value in unique_hs:
        df2 = df[df['hedging_set'] == hs_value].copy()
        df2 = df2.sort_values(by='mat_bucket')

        df2['D1'] = df2.loc[df2['mat_bucket'] == 1, 'D']
        df2['D2'] = df2.loc[df2['mat_bucket'] == 2, 'D']
        df2['D3'] = df2.loc[df2['mat_bucket'] == 3, 'D']

        df2 = df2.groupby(['counterparty', 'product', 'rate', 'Supervisory_factor', 'hedging_set', 'RC']).agg({'D1': 'sum', 'D2': 'sum', 'D3': 'sum', 'uncoll_mtm': 'sum'}).reset_index()
        #to find effective total notional for each hedging set
        df2['effective_notional'] = (df2['D1']**2 + df2['D2']**2 + df2['D3']**2 + 1.4 * df2['D1'] * df2['D2'] + 1.4 * df2['D2'] * df2['D3'] + 0.6 * df2['D1'] * df2['D3'])**(1/2)

        data = pd.concat([data, df2])
    data = data.groupby(['counterparty', 'rate', 'RC']).apply(lambda x:
                                                                  pd.Series({
                                                                      'PFE': (x['Supervisory_factor'] * x[
                                                                          'effective_notional']).sum(),
                                                                      'uncoll_mtm': (x['uncoll_mtm']).sum()
                                                                  })).reset_index()
    return data

def multiplier(deals_grouped, dealsCDS):
    # to add credit deals if any
    deals_grouped = pd.concat([deals_grouped, dealsCDS])
    #to group current exposure and PFE by every counterparty
    deals_grouped = deals_grouped.groupby(['counterparty', 'RC']).agg(
        {'PFE': 'sum', 'uncoll_mtm': 'sum'}).reset_index()
    deals_grouped['multiplier'] = np.minimum(
        1, 0.05 + 0.95 * np.exp(deals_grouped['uncoll_mtm'] / (2 * 0.95 * deals_grouped['PFE'])))
    return deals_grouped

def EAD(deals_grouped):
    deals_grouped.loc[deals_grouped['uncoll_mtm'] < 0, 'uncoll_mtm'] = 0
    deals_grouped['EAD'] = (1.4 * (deals_grouped['RC'] + deals_grouped['multiplier'] * deals_grouped['PFE']))
    return deals_grouped

def credit_koef_calc(deals):
    deals = maturity_factor_calc(deals)
    #вставить цикл

    dealsCDS = deals.groupby(['counterparty', 'product', 'Class', 'ccy_1', 'Supervisory_factor', 'Correlation', 'RC']).apply(
        lambda x:
        pd.Series({
            'effective_notional': (x['sdelta'] * x['adj_notional'] * x['maturity_factor']).sum(),
            'uncoll_mtm': (x['mtm'] - x['collateral']).sum()
        })).reset_index()
    dealsCDS['entity_level_addon'] = dealsCDS['Supervisory_factor'] * dealsCDS['effective_notional']
    dealsCDS['systematic_component'] = (dealsCDS['Correlation'] * dealsCDS['entity_level_addon'])
    dealsCDS['idiosyncratic_component'] = (1-(dealsCDS['Correlation']) ** 2) * (dealsCDS['entity_level_addon']) ** 2
    dealsCDS = dealsCDS.groupby(['counterparty', 'RC', 'Class'], as_index=False).agg({'uncoll_mtm': 'sum', 'systematic_component': 'sum', 'idiosyncratic_component': 'sum'}).reset_index()
    dealsCDS['PFE'] = (dealsCDS['systematic_component'] ** 2 + dealsCDS['idiosyncratic_component']) ** (1 / 2)

    dealsCDS = dealsCDS.groupby(['counterparty', 'RC'], as_index=False).agg({'PFE': 'sum', 'uncoll_mtm': 'sum'}).reset_index()

    return dealsCDS

def divide(deals):
    dealsCDS = deals[deals['product'].isin(['CD_ААА', 'CD_AA', 'CD_A', 'CD_BBB',
                                            'CD_BB', 'CD_B', 'CD_CCC', 'CI_IG',
                                            'CI_SG', 'E', 'E_Index', 'CMD_Electricity',
                                            'CMD_ Oil/Gas', 'CMD_ Metals',
                                            'CMD_ Agricultural', 'CMD_ Other',])].copy()
    deals = deals[~(deals['product'].isin(['CD_ААА', 'CD_AA', 'CD_A', 'CD_BBB',
                                            'CD_BB', 'CD_B', 'CD_CCC', 'CI_IG',
                                            'CI_SG', 'E', 'E_Index', 'CMD_Electricity',
                                            'CMD_ Oil/Gas', 'CMD_ Metals',
                                            'CMD_ Agricultural', 'CMD_ Other',]))].copy()
    return deals, dealsCDS

deals = pd.read_excel('/Users/mihailzaytsev/Desktop/CCR/SA_CCR/deals/SA6.xlsx', engine='openpyxl')
supervisory_parameters = pd.read_excel('/Users/mihailzaytsev/Desktop/practical-python/Risk/Supervisory_parametres.xlsx', engine='openpyxl')
deals = deals.merge(supervisory_parameters[['Asset', 'Class', 'Supervisory_factor', 'Correlation', 'Supervisory_option_volatility']],
                    how='left', left_on='product', right_on='Asset').drop('Asset', axis=1)

deals[['year_frac_s', 'year_frac_e', 'year_frac_m']] = deals[['year_frac_s', 'year_frac_e',	'year_frac_m']].astype(float)

deals = RC(deals)
deals = sduration_calc(deals)
deals = adj_notional_calc(deals)
deals = sdelta_calc(deals)
deals, dealsCDS = divide(deals)
if not dealsCDS.empty:
    dealsCDS = credit_koef_calc(dealsCDS)
deals = maturity_factor_calc(deals)
deals = maturity_bucket(deals)
deals = hedging_sets_calc(deals)
deals = multiplier(deals, dealsCDS)
deals = EAD(deals)

print(deals)
