from scipy.special import expit
import numpy as np
import math


# Simulate Hidden Polls
def sample_hidden_poll(num_voters, num_ts, avg_ava):
    return np.random.binomial(n=1, p=avg_ava, size=(num_voters, num_ts))

# Simulate individual utilities
def generate_individual_utils(num_voters, num_ts, hidden_poll, tau_a1=0.8):
    u_mat = np.zeros((num_voters, num_ts))
    a1_utils = np.random.uniform(low=tau_a1, high=1, size=u_mat.shape)
    u_mat[np.where(hidden_poll==1)] = a1_utils[np.where(hidden_poll==1)]
    a23_utils = np.random.uniform(low=0, high=tau_a1, size=u_mat.shape)
    u_mat[np.where(hidden_poll==0)] = a23_utils[np.where(hidden_poll==0)]
    return u_mat


def calc_max_soc_util(tau_a1=0.8, tau_a2=0.4):
    return tau_a1 - tau_a2


# Calculate social utilities and simulate open poll
def calc_open_poll(num_voters, num_ts, u_mat, max_soc_util, tau_a1=0.8, tau_a2=0.4):
    socio_mat, open_poll = np.zeros((num_voters, num_ts)), np.zeros((num_voters, num_ts)).astype(int)
    tao_soc = max_soc_util + tau_a2
    for voter in range(num_voters):
        for ts in range(num_ts):
            pop = np.mean(open_poll[:voter,ts])
            socio_mat[voter,ts] = calc_su(pop, voter, num_voters, max_soc_util)
            tot_u = u_mat[voter,ts] + socio_mat[voter,ts]
            if (u_mat[voter,ts] > tau_a1) or (tot_u > tao_soc):
                open_poll[voter,ts] = 1
    return open_poll


# Social utility modeling
def su_pop(pop, mu, nat_pop=0.5):    
    if pop <=nat_pop:
        y = mu*(1-expit(16*pop-4.5))
    else:
        y = mu+(-mu*(1-expit(20*(pop-nat_pop)-4.5)))
    return y


def su_n(n, N):
    return expit(10*((n/N))-2)


def calc_su(pop, n, num_voters, max_su, nat_pop=0.5):
    if nat_pop != 0.5:
        raise NotImplementedError
    if math.isnan(pop):
        return 0
    else:
        su = (su_pop(pop, max_su, nat_pop)) * su_n(n, num_voters)
        return su
    

def calc_su_hat(pop, n, num_voters, max_su, nat_pop=0.5):
    if nat_pop != 0.5:
        raise NotImplementedError
    if math.isnan(pop):
        return 0
    else:
        su = (su_pop_hat2(pop, max_su, nat_pop)) * su_n(n, num_voters)
        return su
    
# estimate social utils based on essumed max social util value
def estimate_social_utils(open_poll, num_voters, num_ts, max_soc_util=1):
    socio_mat_hat = np.zeros((num_voters, num_ts))
    for voter in range(num_voters):
        for ts in range(num_ts):
            pop = np.mean(open_poll[:voter,ts])
            socio_mat_hat[voter,ts] = calc_su(pop, voter, num_voters, max_soc_util, nat_pop=0.5)
    return socio_mat_hat


def calc_weights(socio_mat_hat, max_soc_util):
    weights = 1 - 1*socio_mat_hat
    weights[0, :] = np.max(weights[1:,:])
    return weights


def declare_winner(weights, open_poll):
    max_score = np.max(weights.sum(axis=0))
    winner_dda = np.argmax(np.sum(open_poll, axis=0))
    score_dda = weights.sum(axis=0)[winner_dda]
    winner_weights = np.argmax(np.sum(weights*open_poll, axis=0))

    if max_score - score_dda > 0.1*max_score: #and (open_poll.shape[0]>= 12) and (open_poll.shape[1] >= 10):
        final_winner = winner_weights
    else:
        final_winner = winner_dda
    return winner_dda, winner_weights, final_winner, score_dda, max_score


def update_res_dict(res_dict, num_voters, num_ts, tau_a1, tau_a2, max_soc_util, 
                    winner_results, trial_num, util_dda, util_weights, open_ava, hidden_ava, hidden_dda):
    winner_dda, winner_weights, final_winner, score_dda, max_score = winner_results
    res_dict['num_voters'].extend([num_voters])
    res_dict['num_ts'].extend([num_ts])
    res_dict['tau_a1'].extend([tau_a1])
    res_dict['tau_a2'].extend([tau_a2])
    res_dict['max_soc_util'].extend([max_soc_util])
    res_dict['winner_dda'].extend([winner_dda])
    res_dict['winner_weights'].extend([winner_weights])
    res_dict['final_winner'].extend([final_winner])
    res_dict['score_dda'].extend([score_dda])
    res_dict['max_score'].extend([max_score])
    res_dict['util_dda'].extend([util_dda])
    res_dict['util_weights'].extend([util_weights])
    res_dict['trial_num'].extend([trial_num])
    res_dict['open_ava'].extend([open_ava])
    res_dict['hidden_ava'].extend([hidden_ava])
    res_dict['hidden_winner'].extend([hidden_dda])

    assert open_ava >= hidden_ava
    return res_dict


def calc_confusion_mat(df):
    a00 = np.logical_and(df["winner_dda"] != df["hidden_winner"], df["final_winner"] != df["hidden_winner"]).sum()
    a01 = np.logical_and(df["winner_dda"] == df["hidden_winner"], df["final_winner"] != df["hidden_winner"]).sum()
    a10 = np.logical_and(df["winner_dda"] != df["hidden_winner"], df["final_winner"] == df["hidden_winner"]).sum()
    a11 = np.logical_and(df["winner_dda"] == df["hidden_winner"], df["final_winner"] == df["hidden_winner"]).sum()
    return np.array([[a00,a01],[a10,a11]])