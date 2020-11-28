import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings

from matplotlib_venn import venn3
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from statsmodels.stats.proportion import proportion_confint
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# turn off the annoying warnings
pd.options.mode.chained_assignment = None
warnings.simplefilter('ignore', ConvergenceWarning)


def main():
    # find adopted users and prepare features
    users = get_complete_user_data()

    x = users.drop('adopted', axis=1)
    y = users['adopted']

    # model with usage period
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    print('F-1 Score with usage data:', f1_score(y_test, LogisticRegression().fit(x_train, y_train).predict(x_test)))

    # drop usage period as a practically useless feature
    users.drop('usage_period_ts', axis=1, inplace=True)
    x.drop('usage_period_ts', axis=1, inplace=True)

    # correlation matrix as F-1 scores (since all variables are booleans)
    print(f1_score_matrix(users).to_string())

    # try again without data on usage
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    model = get_tuned_model(x_train, y_train)

    max_score = plot_cutoffs(model, x_test, y_test, min_cutoff=0.01, max_cutoff=0.2, step=0.01)
    print('Max F-1 Score:', max_score)

    plot_users_venns(users)
    plot_adoption_sources(users)

    # rate of adoption given [feature] compared to base rate of adoption (~13%)
    influences = get_influences(users)
    print(influences)
    plot_influence(influences)

    importances = get_gini_importances(x, y)
    print(importances)

    logit_model = sm.Logit(y, x).fit()
    print(get_summary_df(importances, influences, logit_model).to_string())


def get_complete_user_data():
    user_engagement = pd.read_csv('data/user_engagement.csv', encoding='windows-1252')
    users = pd.read_csv('data/users.csv', encoding='windows-1252').rename({'object_id': 'user_id'}, axis=1)

    user_engagement['time_stamp'] = pd.DatetimeIndex(user_engagement['time_stamp'])

    # table of bools of whether a given user logged in on a given day, of format:
    #                   2019-01-01  2019-01-02  ...
    #             user1     True        False
    #             user2     False       False
    user_engagement = (user_engagement.groupby(by=['user_id',  # group by days to normalize timestamps (13:44 -> 00:00)
                                                   pd.Grouper(key='time_stamp', freq='1D')])
                       ['visited'].count()  # count all visits by day (in case somebody logged in several times per day)
                       .reset_index().pivot(index='user_id',  # day1 day2 day3
                                            columns='time_stamp',  # user1  1   nan   1
                                            values='visited')  # user2 nan  nan   1
                       .reindex(pd.date_range(user_engagement['time_stamp'].min().normalize(),  # add days that
                                              user_engagement['time_stamp'].max().normalize()),  # nobody logged in on
                                axis=1,
                                fill_value=0).fillna(0).astype(bool))

    adopted_users = (user_engagement.transpose()  # transpose() works better than axis=1
                     .rolling(window=7).sum() >= 3  # logged in on at least 3 different days within one 7-day window
                     ).any().replace(False, np.nan).dropna().index.values

    users.index = users['user_id']  # duplicating users_id column as index to access eay with .loc

    users['adopted'] = False
    users['adopted'][users['user_id'].isin(adopted_users)] = True

    users['last_login_date_dt'] = user_engagement.sort_index(axis=1, ascending=False).idxmax(axis=1)
    users['first_login_date_dt'] = user_engagement.idxmax(axis=1)
    users['usage_period_dt'] = (users['last_login_date_dt'] - users['first_login_date_dt'])

    # convert to timestamps for easier processing
    users['last_login_date_ts'] = users['last_login_date_dt'].values.astype('int64') // 1e9
    # nans in int64 are -9223372037, so need to convert them back to nans
    users['last_login_date_ts'][users['last_login_date_dt'].isna()] = np.nan

    users['first_login_date_ts'] = users['first_login_date_dt'].values.astype('int64') // 1e9
    users['first_login_date_ts'][users['first_login_date_dt'].isna()] = np.nan

    users['usage_period_ts'] = users['last_login_date_ts'] - users['first_login_date_ts']

    users['last_session_creation_time_dt'] = pd.to_datetime(users['last_session_creation_time'] * 1e9)
    users['creation_time_dt'] = pd.to_datetime(users['creation_time'])
    users['creation_time'] = users['creation_time_dt'].values.astype('int64') // 1e9

    users['invited_by_user'] = ~users['invited_by_user_id'].isna()  # new bool column (invited or not)

    users['opted_in_to_mailing_list'] = users['opted_in_to_mailing_list'].astype(bool)
    users['enabled_for_marketing_drip'] = users['enabled_for_marketing_drip'].astype(bool)

    # convert column 'creation_source' with 5 categories into 5 bool columns for each of the sources
    users = pd.concat([users, pd.get_dummies(users['creation_source']).astype(bool)], axis=1)

    invitees = users[users['invited_by_user']]
    inviters = users.loc[invitees['invited_by_user_id'].astype(int)]

    users['inviter_adopted'] = False
    users.loc[invitees.index, 'inviter_adopted'] = inviters['adopted'].values

    # seems to be highly correlated with 'invited_by_user', so will just not include it
    # users['inviter_same_email_domain'] = False
    # users.loc[invitees.index,
    #           'inviter_same_email_domain'] = invitees['email_domain'].values != inviters['email_domain'].values

    # creation source of inviter, also seems to be useless
    users['inviter_creation_source'] = np.nan
    # users.loc[invitees.index, 'inviter_creation_source'] = inviters['creation_source'].values
    # users = pd.concat([users, pd.get_dummies(users['inviter_creation_source'], prefix='inviter').astype(bool)], axis=1)

    # all people who received an invite, were invited by a person in their organization, so not a useful metric
    assert (invitees['org_id'].values == inviters['org_id'].values).all()

    # normalize usage period
    users['usage_period_ts'] = preprocessing.scale(users['usage_period_ts'].fillna(0))

    # drop non-feature columns
    users.drop(['last_session_creation_time', 'creation_source', 'usage_period_dt', 'email_domain',
                'invited_by_user_id', 'org_id', 'last_login_date_ts', 'last_login_date_dt', 'creation_time',
                'first_login_date_dt', 'user_id', 'creation_time_dt', 'last_session_creation_time_dt', 'name', 'email',
                'first_login_date_ts', 'inviter_creation_source'], axis=1, inplace=True)

    return users


def get_tuned_model(x_train, y_train):
    grid = {
        "n_estimators": [10, 50, 100, 150],
        "max_depth": [3, 4, 6, 10, 20],
    }
    grid_search = GridSearchCV(RandomForestClassifier(), grid, n_jobs=-1, cv=5)
    grid_search.fit(x_train, y_train)

    forest = RandomForestClassifier(**grid_search.best_params_).fit(x_train, y_train)

    return forest


def get_gini_importances(x, y):
    forest = RandomForestClassifier().fit(x, y)

    return pd.DataFrame([x.columns, forest.feature_importances_],
                        index=['Feature', 'Gini importance']
                        ).transpose().sort_values('Gini importance', ascending=False)


def get_influences(users):
    mid_rates = sorted_single_var_matrix(users, 'adopted', scorer=influence).rename('Mean Influence')
    low_cis = sorted_single_var_matrix(users, 'adopted', scorer=influence_ci_low).rename('Low CI')
    high_cis = sorted_single_var_matrix(users, 'adopted', scorer=influence_ci_high).rename('High CI')
    return pd.concat([mid_rates, low_cis, high_cis], axis=1)


def sorted_single_var_matrix(df, target_col, **kwargs):
    return custom_matrix(df, target_columns=[target_col], **kwargs).sort_values(target_col)[target_col].iloc[-2::-1]


def custom_matrix(df, scorer=f1_score, target_columns=()):
    if len(target_columns) == 0:
        target_columns = df.columns
    result = pd.DataFrame(data=np.nan, index=df.columns, columns=df.columns)
    for column_true in target_columns:
        for column_pred in df.columns:
            # non-na df
            clean_df = df[{column_true, column_pred}].dropna()
            try:
                score = scorer(clean_df[column_true], clean_df[column_pred])
            except ValueError:
                continue
            result.loc[column_pred, column_true] = score

    return result[result.columns[~result.isna().all()]]  # drop columns with all nans


def f1_score_matrix(df, **kwargs):
    return custom_matrix(df, scorer=f1_score, **kwargs)


def influence(y_true, y_pred):
    [not_true_not_pred, not_true_pred], [true_not_pred, true_pred] = confusion_matrix(y_true, y_pred)
    base_rate = (true_not_pred + true_pred) / (not_true_not_pred + not_true_pred + true_not_pred + true_pred)
    pred_rate = true_pred / (true_pred + not_true_pred)
    return pred_rate / base_rate


def influence_ci(y_true, y_pred, ci='both'):
    """
    :param ci:
        if 'both': returns a tuple (ci_low, ci_high)
        if 'low': returns only low confidence interval
        if 'high': returns only high confidence interval
    """
    [not_true_not_pred, not_true_pred], [true_not_pred, true_pred] = confusion_matrix(y_true, y_pred)
    base_rate = (true_not_pred + true_pred) / (not_true_not_pred + not_true_pred + true_not_pred + true_pred)
    ci_low, ci_high = proportion_confint(true_pred, (true_pred + not_true_pred))
    if ci == 'low':
        return ci_low / base_rate
    elif ci == 'high':
        return ci_high / base_rate
    else:
        return ci_low, ci_high / base_rate


def influence_ci_low(y_true, y_pred):
    return influence_ci(y_true, y_pred, ci='low')


def influence_ci_high(y_true, y_pred):
    return influence_ci(y_true, y_pred, ci='high')


def influence_matrix(df, **kwargs):
    return custom_matrix(df, scorer=influence, **kwargs)


def cond_prob(y_true, y_pred):
    [_, not_true_pred], [_, true_pred] = confusion_matrix(y_true, y_pred)
    return true_pred / (true_pred + not_true_pred)


def cond_prob_matrix(df, **kwargs):
    return custom_matrix(df, scorer=cond_prob, **kwargs)


def get_predictions(model, x, cutoff=0.5):
    return model.predict_proba(x).transpose()[1] >= cutoff


def plot_cutoffs(model, x_test, y_test, scorer=f1_score, min_cutoff=0.1, max_cutoff=0.5, step=0.1):
    scores = []
    cutoffs = np.arange(min_cutoff, max_cutoff + step, step)
    for cutoff in cutoffs:
        scores.append(scorer(y_test, get_predictions(model, x_test, cutoff=cutoff)))
    plt.plot(cutoffs, scores)
    plt.title('Score depending on the confidence cutoff of ' + type(model).__name__)
    plt.xlabel('Confidence cutoff')
    plt.ylabel(scorer.__name__)
    plt.show()
    return max(scores)


def plot_adoption_sources(users):
    adoption_sources = users[users['adopted']].drop(
        ['opted_in_to_mailing_list', 'enabled_for_marketing_drip', 'adopted', 'invited_by_user', 'inviter_adopted'],
        axis=1).sum().sort_values(ascending=False) / len(users[users['adopted']])
    plot_donut(adoption_sources, adoption_sources.index)
    plt.savefig('dataviz/adoption_sources.png', dpi=100)
    plt.show()


def plot_influence(influences):
    influences_percent = influences * 100
    errors = influences_percent['Mean Influence'] - influences_percent['Low CI']
    means = influences_percent['Mean Influence']
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(influences_percent.index, means, yerr=errors, color='magenta')
    plt.xticks(rotation=-45, ha="left", rotation_mode="anchor")
    plt.axhline(100, linestyle='dotted')
    plt.ylim(0, 200)

    ax.annotate('Positive impact', xy=(len(influences) // 2, 150), size=32, color='grey', alpha=0.4,
                ha='center', va='center')
    ax.annotate('Negative impact', xy=(len(influences) // 2, 50), size=32, color='grey', alpha=0.4,
                ha='center', va='center')
    plt.title('Influence of given factors on the rate of adoption')
    plt.ylabel('Influence on rate of adoption (%)')
    plt.savefig('dataviz/influence.png', dpi=100)
    plt.show()


def plot_users_venns(users):
    fig, [[ax_top_left, ax_top_right], [ax_bottom_left, ax_bottom_right]] = plt.subplots(2, 2, figsize=(7.5, 5))
    plot_venn(pd.concat([users, pd.Series(True, index=users.index, name='all_users')], axis=1),
              ['adopted', 'invited_by_user', 'all_users'], ax=ax_top_left)
    plot_venn(users, ['adopted', 'inviter_adopted', 'invited_by_user'], ax=ax_top_right)
    plot_venn(users, ['adopted', 'opted_in_to_mailing_list', 'enabled_for_marketing_drip'], ax=ax_bottom_right)
    plot_venn(users, ['adopted', 'inviter_adopted', 'ORG_INVITE'], ax=ax_bottom_left)

    fig.suptitle('Overlap in user data')
    fig.savefig('dataviz/venns.png', dpi=100)
    plt.show()


def plot_decision_tree(tree, predictors_cols):
    fig, ax = plt.subplots(figsize=(50, 24))
    plot_tree(tree, fontsize=6, feature_names=predictors_cols)
    plt.show()


def plot_donut(data, labels):
    # code borrowed from https://matplotlib.org/3.1.1/gallery/pie_and_polar_charts/pie_and_donut_labels.html
    fig, ax = plt.subplots(figsize=(8, 4), subplot_kw=dict(aspect="equal"))

    wedges, texts = ax.pie(data, wedgeprops=dict(width=0.5), startangle=90, counterclock=False, normalize=False)

    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle="-"),
              bbox=bbox_props, zorder=0, va="center")

    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1) / 2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate(labels[i], xy=(x, y), xytext=(1.35 * np.sign(x), 1.4 * y),
                    horizontalalignment=horizontalalignment, **kw)


def plot_venn(df, columns, ax=plt.gca(), **kwargs):
    return venn3([
        set(df.index[df[columns[0]]]),
        set(df.index[df[columns[1]]]),
        set(df.index[df[columns[2]]])
    ], columns, ax=ax, **kwargs)


def get_summary_df(importances, influences, logit_model):
    return pd.concat(
        [influences['Mean Influence'], (influences['Mean Influence'] - influences['Low CI']).rename('Influence error'),
         importances.set_index('Feature'), logit_model.params.rename('Logit coeficients'),
         logit_model.pvalues.rename('Logit p-values')], axis=1)


if __name__ == '__main__':
    main()
