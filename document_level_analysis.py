"""
/src/local_geometry_analysis.py

description:
    This script reads in two topic-feature matrices and analyzes how the ranking of each topic's nearest
    neighbors changes between them using Kendall's tau. While CKA describes global geometric differences
    between two matrices, this analysis describes local geometric differences for each topic.

"""
import numpy as np
from scipy.spatial.distance import cdist, cosine
from scipy.stats import kendalltau, pearsonr, spearmanr
from pathlib import Path
import os
from itertools import combinations
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

def neighbor_rankings(X, Y, metric="cosine"):
    """
    """
    assert X.shape[0] == Y.shape[0]
    n = X.shape[0]
    
    # compute cosine distance matrices within each space
    DX = cdist(X, X, metric=metric)
    DY = cdist(Y, Y, metric=metric)

    rankings_x = []
    rankings_y = []

    for i in range(n):
        neighbors = np.arange(n) != i
        neighbor_indices = np.arange(n)[neighbors]

        order_x = neighbor_indices[np.argsort(DX[i, neighbors])]
        order_y = neighbor_indices[np.argsort(DY[i, neighbors])]

        rankings_x.append(order_x)
        rankings_y.append(order_y)

    return rankings_x, rankings_y


def kendalls_tau(X, Y, metric="cosine"):
    """

    Params:
    -------
    X : ndarray, shape (n_samples, d1)
    Y : ndarray, shape (n_samples, d2)
        Row i in X must already correspond to row i in Y.
    metric: str
        Distance metric from scipy.spatial.distance.cdist

    Returns
    -------
    tau_list : ndarray, shape (n_samples)
        Kendall's tau for each row. Element i in tau_list corresponds to row i in X and Y.
    rankings_x : ndarray
    rankings_y : ndarray
    """
    
    assert X.shape[0] == Y.shape[0]

    rankings_x, rankings_y = neighbor_rankings(X, Y)

    n = X.shape[0]

    tau_list = np.empty(n)

    for i in range(n):

        # convert neighbor orderings to rank vectors
        rank_x = np.empty(n - 1, dtype=int)
        rank_y = np.empty(n - 1, dtype=int)

        rank_x[rankings_x[i].argsort()] = np.arange(n - 1)
        rank_y[rankings_y[i].argsort()] = np.arange(n - 1)

        tau_list[i] = kendalltau(rank_x, rank_y).statistic

        rank_x_pos = {doc_idx: rank for rank, doc_idx in enumerate(rankings_x[i])}
        rank_y_pos = {doc_idx: rank for rank, doc_idx in enumerate(rankings_y[i])}
        
    return tau_list, rankings_x, rankings_y 


def excerpt_and_summary_rank_analysis():
    """
    X:
    ---
    ├── excerpts
    │   ├── Qwen3.5-0.8B
    │   │   ├── document_embeddings_merged_agnostic.npy

    Y:
    ---
    └── summaries
        ├── Qwen3.5-0.8B
        │   ├── document_embeddings_merged_agnostic.npy

    """
    qwen_exc_path = os.path.join(PROJECT_ROOT, 'data', 'run12', 'excerpts', 'Qwen3.5-0.8B', 'document_embeddings_merged_agnostic.npy')
    qwen_sum_path = os.path.join(PROJECT_ROOT, 'data', 'run12', 'summaries', 'Qwen3.5-0.8B', 'document_embeddings_merged_agnostic.npy')
    qwen_exp_path = os.path.join(PROJECT_ROOT, 'data', 'run12', 'responses', 'Qwen3.5-0.8B', 'document_embeddings_merged_agnostic.npy')
    qwen_tau_list = kendalls_tau_analysis_by_document(qwen_exc_path, qwen_sum_path, 'Qwen-run-12')
    #qwen_tau_list = kendalls_tau_analysis_by_document(qwen_exc_path, qwen_exp_path, 'Qwen-run-12_explanations')
 
    gemm_exc_path = os.path.join(PROJECT_ROOT, 'data', 'run12', 'excerpts', 'gemma-3-1b-it', 'document_embeddings_merged_agnostic.npy')
    gemm_sum_path = os.path.join(PROJECT_ROOT, 'data', 'run12', 'summaries', 'gemma-3-1b-it', 'document_embeddings_merged_agnostic.npy')
    gemm_exp_path = os.path.join(PROJECT_ROOT, 'data', 'run12', 'responses', 'gemma-3-1b-it', 'document_embeddings_merged_agnostic.npy')
    gemm_tau_list = kendalls_tau_analysis_by_document(gemm_exc_path, gemm_sum_path, 'gemma-run-12')
    #gemm_tau_list = kendalls_tau_analysis_by_document(gemm_exc_path, gemm_exp_path, 'gemma-run-12_explanations')

    # TO DO (as of 4:11 pm on Monday July 27)
    # - make plot of excerpt[token_count] - summary[token_count] as x-axis and Kendall's tau as y-axis for both qwen and gemma.
    
    # plot of token length diffs against kendall's tau:
    qwen_excerpt_data = np.load(qwen_exc_path, allow_pickle=True).item()
    qwen_summary_data = np.load(qwen_sum_path, allow_pickle=True).item()

    qwen_excerpt_meta = qwen_excerpt_data['meta']
    qwen_summary_meta = qwen_summary_data['meta']
    
    qwen_token_len_excerpts = []
    qwen_token_len_diffs = []
    for i in range(len(qwen_excerpt_meta)):
        #print(qwen_excerpt_meta[i]['token_count'], qwen_summary_meta[i]['token_count'], qwen_excerpt_meta[i]['token_count'] - qwen_summary_meta[i]['token_count'])
        qwen_token_len_diffs.append(qwen_excerpt_meta[i]['token_count'] - qwen_summary_meta[i]['token_count'])
        qwen_token_len_excerpts.append(qwen_excerpt_meta[i]['token_count'])
    #taus_vs_token_len_diffs_plot(qwen_token_len_diffs, qwen_tau_list, 'run12_qwen')
    taus_vs_token_len_plot(qwen_token_len_excerpts, qwen_tau_list, 'run12_qwen') 


    # Same thing for Gemma now: compare exceprt length with kendall's tau.
    gemm_excerpt_data = np.load(gemm_exc_path, allow_pickle=True).item()
    gemm_summary_data = np.load(gemm_sum_path, allow_pickle=True).item()

    gemm_excerpt_meta = gemm_excerpt_data['meta']
    gemm_summary_meta = gemm_summary_data['meta']

    gemm_token_len_excerpts = []
    for i in range(len(gemm_excerpt_meta)):
        gemm_token_len_excerpts.append(gemm_excerpt_meta[i]['token_count'])
    taus_vs_token_len_plot(gemm_token_len_excerpts, gemm_tau_list, 'run12_gemma')


    # Plot of qwen vs gemma tau values:
    #qwen_vs_gemma_taus_plot(qwen_tau_list, gemm_tau_list, 'run12')


def taus_vs_token_len_plot(lengths, taus, name):
    x = np.asarray(lengths)
    y = np.asarray(taus)

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    ax.scatter(x,
               y,
               alpha=0.5,
               marker='o',
               facecolors='none',
               edgecolors='black')
    ax.set_xlabel("Excerpt length")
    ax.set_ylabel("Kendall's $\\tau$")

    plt.tight_layout()
    plt.savefig(name + "_kendalls_tau_vs_excerpt_length.tiff",
                dpi=600,
                bbox_inches="tight")
    plt.savefig(name + "_kendalls_tau_vs_excerpt_length.svg", bbox_inches="tight")
    plt.show()
    plt.close()


def taus_vs_token_len_diffs_plot(diffs, taus, name):
    x = np.asarray(diffs)
    y = np.asarray(taus)
    
    # add in pearsonr and spearmanr here to emphasize lack of correlation. 
    
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    ax.scatter(x,
               y,
               alpha=0.5,
               marker='o',
               facecolors='none',
               edgecolors='black')
    ax.set_xlabel("excerpt length $-$ summary length")
    ax.set_ylabel("Kendall's $\\tau$")

    plt.tight_layout()
    plt.savefig(name + "_kendalls_tau_vs_token_difference.tiff",
                dpi=600,
                bbox_inches="tight")
    plt.savefig(name + "_kendalls_tau_vs_token_difference.svg", bbox_inches="tight")
    plt.show()
    plt.close()


def qwen_vs_gemma_taus_plot(qw_taus, ge_taus, name):
    x = np.asarray(ge_taus)
    y = np.asarray(qw_taus)

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    ax.scatter(x,
               y,
               s=18,
               alpha=0.4,
               marker='o',
               facecolors='none',
               edgecolors='black')

    ax.set_xlim(-0.25, 0.25)
    ax.set_ylim(-0.25, 0.25)
    ax.set_xlabel("Kendall's tau from gemma")
    ax.set_ylabel("Kendall's tau from qwen")

    plt.tight_layout()
    plt.savefig(name + "_kendalls_tau_qwen_vs_gemma.tiff",
                dpi=600,
                bbox_inches="tight")
    plt.savefig(name + "kendalls_tau_qwen_vs_gemma.svg", bbox_inches="tight")
    plt.show()
    plt.close()


def kendalls_tau_analysis_by_document(exc_path, sum_path, analysis_name):
    """
    """
    print(analysis_name)

    excerpt_data = np.load(exc_path, allow_pickle=True).item()
    summary_data = np.load(sum_path, allow_pickle=True).item()

    excerpt_matrix = excerpt_data['embeddings']
    summary_matrix = summary_data['embeddings']

    print('size of embedding matrices:', excerpt_matrix.shape, summary_matrix.shape)

    excerpt_meta = excerpt_data['meta']
    summary_meta = summary_data['meta']

    #print('shape of excerpt_matrix', excerpt_matrix.shape)
    #print('shape of summary_matrix', summary_matrix.shape)

    tau_list, rankings_exc, rankings_sum = kendalls_tau(excerpt_matrix, summary_matrix)
    print('kendall tau statistics:')
    print('------------------------')
    print('mean: ', np.mean(tau_list))
    print('std: ', np.std(tau_list))
    print('proportion negative: ', np.mean(tau_list < 0.0))
    print('propotion positive: ', np.mean(tau_list > 0.0))
    print('-------------------------')
    sorted_tau_indices = np.argsort(tau_list)
    length_list = [row['token_count'] for row in excerpt_meta]
    pear_coef, pear_p = pearsonr(tau_list, length_list)
    print('pearson coef:', pear_coef, 'pvalue=', pear_p)
    print()
    
    for rank, idx in enumerate(sorted_tau_indices):
        print(rank, '->', idx, 'title', excerpt_meta[idx]['title'], 'tau: ', tau_list[idx]) 
    
    
    most_changed_indices = sorted_tau_indices[:10]
    least_changed_indices = sorted_tau_indices[-10:]
    print()
    print('*****************')
    print('least changed:')
    for i, idx in enumerate(sorted_tau_indices[-10:]):
        print(i+1, '.', 'doc_id:', excerpt_meta[idx]['doc_id'], 'title:', excerpt_meta[idx]['title'], 'tau:', round(tau_list[idx], 4), 'category:', excerpt_meta[idx]['category'], 'legnth', excerpt_meta[idx]['token_count'])
    #print(least_changed_indices)
    print()
    print('most changed:')
    for i, idx, in enumerate(sorted_tau_indices[:10]):
        print(i+1, '.', 'doc_id:', excerpt_meta[idx]['doc_id'], 'title:', excerpt_meta[idx]['title'], 'tau:', round(tau_list[idx], 4), 'category:', excerpt_meta[idx]['category'], 'length', excerpt_meta[idx]['token_count'])
    #print(most_changed_indices)
    print('*****************')
    print()

    # for the most changed docs, do the following
    #   1) print the excerpt text and the summary text. 
    #   2) get the 10 nearest neighbors from within the excerpt embeddings 
    #   3) get the 10 neareast neighbors from within the summary embeddings
    #   4) get the discordant pairs
    '''
    # most changed: most_changed_indices[0]
    case_1_idx = sorted_tau_indices[0]
    case_1_exc_meta = excerpt_meta[case_1_idx]
    case_1_sum_meta = summary_meta[case_1_idx]

    print('most changed doc ->', 'row=', case_1_idx, '; tau=', tau_list[case_1_idx])
    print('*'*8)
    print('metadata from excerpts:')
    print('doc_id:', case_1_exc_meta['doc_id'])
    print('title:', case_1_exc_meta['title'])
    print('token_count:', case_1_exc_meta['token_count'])
    print('*'*8)
    print('metadata from summaries:')
    print('*'*8)
    print('doc_id:', case_1_sum_meta['doc_id'])
    print('title:', case_1_sum_meta['title'])
    print('token_count:', case_1_sum_meta['token_count'])
    print('*'*8)
    
    # let's look at the 10 nearest neighbors
    print('nearest neighbors in excerpt embeddings:')
    for i, nb_idx in enumerate(rankings_exc[case_1_idx][:10]):
        print('neighbor index:', nb_idx, 'title:', excerpt_meta[nb_idx]['title'], '|  category:', excerpt_meta[nb_idx]['category'], 'token_count:', excerpt_meta[nb_idx]['token_count'])
    print('*'*8)
    print()
    print('nearest neighbors in summary embeddings:')
    for i, nb_idx in enumerate(rankings_sum[case_1_idx][:10]):
        print('neighbor index:', nb_idx, 'title:', summary_meta[nb_idx]['title'], '|  category:', summary_meta[nb_idx]['category'], 'token_count:', summary_meta[nb_idx]['token_count'])
    
    print('\n')

    # Set of top 30 nearest neighbors:
    excerpt_nearest_docs = set(nb_idx for nb_idx in rankings_exc[case_1_idx][:30])
    excerpt_furthest_docs = set(nb_idx for nb_idx in rankings_exc[case_1_idx][-30:])
    summary_nearest_docs = set(nb_idx for nb_idx in rankings_sum[case_1_idx][:30])
    summary_furthest_docs = set(nb_idx for nb_idx in rankings_sum[case_1_idx][-30:])

    common_nearest = excerpt_nearest_docs & summary_nearest_docs
    exc_only_nearest = excerpt_nearest_docs - summary_nearest_docs
    sum_only_nearest = summary_nearest_docs - excerpt_nearest_docs

    common_furthest = excerpt_furthest_docs & summary_furthest_docs
    exc_only_furthest = excerpt_furthest_docs - summary_furthest_docs
    sum_only_furthest = summary_furthest_docs - excerpt_furthest_docs

    print('Of the 30 nearest neighbors...')
    print(' The following are common to both excerpts and summaries:')
    for nb_idx in common_nearest:
        print('nbidx:', nb_idx, 'title:', excerpt_meta[nb_idx]['title'])
    print()
    print(' The following are unique to excerpts:')
    for nb_idx in exc_only_nearest:
        print('nbidx:', nb_idx, 'title:', excerpt_meta[nb_idx]['title'])
    print()
    print(' The following are unique to summaries:')
    for nb_idx in sum_only_nearest:
        print('nbidx:', nb_idx, 'title:', summary_meta[nb_idx]['title'])


    print('end of', analysis_name)
    print('*'*50, '\n')'''
    return tau_list


def doc_position_shift():

    qwen_exc_path = os.path.join(PROJECT_ROOT, 'data', 'run12', 'excerpts', 'Qwen3.5-0.8B', 'document_embeddings_merged_agnostic.npy')
    qwen_sum_path = os.path.join(PROJECT_ROOT, 'data', 'run12', 'summaries', 'Qwen3.5-0.8B', 'document_embeddings_merged_agnostic.npy')
    qwen_exp_path = os.path.join(PROJECT_ROOT, 'data', 'run12', 'responses', 'Qwen3.5-0.8B', 'document_embeddings_merged_agnostic.npy')

    gemm_exc_path = os.path.join(PROJECT_ROOT, 'data', 'run12', 'excerpts', 'gemma-3-1b-it', 'document_embeddings_merged_agnostic.npy')
    gemm_sum_path = os.path.join(PROJECT_ROOT, 'data', 'run12', 'summaries', 'gemma-3-1b-it', 'document_embeddings_merged_agnostic.npy')
    gemm_exp_path = os.path.join(PROJECT_ROOT, 'data', 'run12', 'responses', 'gemma-3-1b-it', 'document_embeddings_merged_agnostic.npy')

    qw_excerpt_data = np.load(qwen_exc_path, allow_pickle=True).item()
    qw_summary_data = np.load(qwen_sum_path, allow_pickle=True).item()
    qw_explana_data = np.load(qwen_exp_path, allow_pickle=True).item()

    qw_excerpt_matrix = qw_excerpt_data['embeddings']
    qw_summary_matrix = qw_summary_data['embeddings']
    qw_explana_matrix = qw_explana_data['embeddings']

    qw_excerpt_meta = qw_excerpt_data['meta']
    qw_summary_meta = qw_summary_data['meta']
    qw_explana_meta = qw_explana_data['meta']
    
    print('qwen analysis:')
    print('*'*8)
    # Comparison of excerpts with summaries.
    # for each topic, calculate the cosine distance between where the topic lives based on the excerpt and where the topic lives based on the summary.
    qw_cosine_distances = []
    print('cosine distance shift from excerpts to summaries:')
    for i, (x, y) in enumerate(zip(qw_excerpt_matrix, qw_summary_matrix)):
        cos_dist = cosine(x, y)
        qw_cosine_distances.append(cos_dist)
        #print(i, qw_excerpt_meta[i]['title'], qw_summary_meta[i]['title'], 'cosine distance:', cos_dist)     
    qw_cosine_distances = np.array(qw_cosine_distances)
    #print('*'*8, '\n')
    print('mean cosine distance:', np.mean(qw_cosine_distances))
    qw_pearson_coef, qw_pearson_pvalue = pearsonr(qw_cosine_distances, [row['token_count'] for row in qw_excerpt_meta])
    print('pearson:', qw_pearson_coef, ', pvalue=', qw_pearson_pvalue)
    print()

    # Comparison of excerpts with explanations.
    '''qw_cosine_distances = []
    print('cosine distance shift from excerpts to explanations')
    for i, (x, y) in enumerate(zip(qw_excerpt_matrix, qw_explana_matrix)):
        cos_dist = cosine(x, y)
        qw_cosine_distances.append(cos_dist)
        #print(i, qw_excerpt_meta[i]['title'], qw_summary_meta[i]['title'], 'cosine distance:', cos_dist)     
    qw_cosine_distances = np.array(qw_cosine_distances)
    #print('*'*8, '\n')
    print('mean cosine distance:', np.mean(qw_cosine_distances))
    qw_pearson_coef, qw_pearson_pvalue = pearsonr(qw_cosine_distances, [row['token_count'] for row in qw_excerpt_meta])
    print('pearson:', qw_pearson_coef, ', pvalue=', qw_pearson_pvalue)
    print()'''

    # First element is that which has the largest cosine distance between excerpt and summary:
    qw_cosdist_sorted_indices = np.argsort(qw_cosine_distances, descending=True)
    for i, doc_idx in enumerate(qw_cosdist_sorted_indices):
        print(i+1, ': ', 'row', doc_idx, ', ', qw_excerpt_meta[doc_idx]['title'], ' || cosine dist:', qw_cosine_distances[doc_idx], 'tokens:', qw_excerpt_meta[doc_idx]['token_count'], 'category', qw_excerpt_meta[doc_idx]['category'])
    print('*'*8, '\n\n')
     
    ge_excerpt_data = np.load(gemm_exc_path, allow_pickle=True).item()
    ge_summary_data = np.load(gemm_sum_path, allow_pickle=True).item()
    ge_explana_data = np.load(gemm_exp_path, allow_pickle=True).item()

    ge_excerpt_matrix = ge_excerpt_data['embeddings']
    ge_summary_matrix = ge_summary_data['embeddings']
    ge_explana_matrix = ge_explana_data['embeddings']

    ge_excerpt_meta = ge_excerpt_data['meta']
    ge_summary_meta = ge_summary_data['meta']
    ge_explana_meta = ge_explana_data['meta']
    
    print('gemma analysis:')
    print('*'*8)
    ge_cosine_distances = []
    print('cosine distance shift from excerpts to summaries:')
    for i, (x, y) in enumerate(zip(ge_excerpt_matrix, ge_summary_matrix)):
        cos_dist = cosine(x, y)
        #print(i, ge_excerpt_meta[i]['title'], ge_summary_meta[i]['title'], '| cosine distance:', cos_dist)
        ge_cosine_distances.append(cos_dist)
    ge_cosine_distances = np.array(ge_cosine_distances)
    print('mean cosine distance:', np.mean(ge_cosine_distances))
    ge_pearson_coef, ge_pearson_pvalue = pearsonr(ge_cosine_distances, [row['token_count'] for row in ge_excerpt_meta])
    print('pearson:', ge_pearson_coef, ', pvalue=', ge_pearson_pvalue)
    print()

    '''ge_cosine_distances = []
    print('cosine distance shift from excerpts to explanations:')
    for i, (x, y) in enumerate(zip(ge_excerpt_matrix, ge_explana_matrix)):
        cos_dist = cosine(x, y)
        #print(i, ge_excerpt_meta[i]['title'], ge_summary_meta[i]['title'], '| cosine distance:', cos_dist)
        ge_cosine_distances.append(cos_dist)
    ge_cosine_distances = np.array(ge_cosine_distances)
    print('mean cosine distance:', np.mean(ge_cosine_distances))
    ge_pearson_coef, ge_pearson_pvalue = pearsonr(ge_cosine_distances, [row['token_count'] for row in ge_excerpt_meta])
    print('pearson:', ge_pearson_coef, ', pvalue=', ge_pearson_pvalue)
    print()'''

    ge_cosdist_sorted_indices = np.argsort(ge_cosine_distances, descending=True)
    for i, doc_idx in enumerate(ge_cosdist_sorted_indices):
        print(i+1, ': ', 'row', doc_idx, ', ', ge_excerpt_meta[doc_idx]['title'], ' || cosine dist:', ge_cosine_distances[doc_idx], 'tokens:', ge_excerpt_meta[doc_idx]['token_count'], 'category', ge_excerpt_meta[doc_idx]['category'])
    print('*'*8, '\n')

    qw_top_shifts = set(qw_cosdist_sorted_indices[:20])
    ge_top_shifts = set(ge_cosdist_sorted_indices[:20])

    qw_ge_intersection = qw_top_shifts & ge_top_shifts
    qw_not_ge = qw_top_shifts - ge_top_shifts
    ge_not_qw = ge_top_shifts - qw_top_shifts
    print('following are from the top 20 topics with the largest shift in the respective model\'s lantent space')
    print('document ids common to qwen and gemma:', [int(x) for x in qw_ge_intersection])
    print('doc ids in qwen but not in gemma:', [int(x) for x in qw_not_ge])
    print('doc ids in gemma but not in qwen:', [int (x) for x in ge_not_qw])


    # make a plot of token count vs cosine distance
    x_qwen = np.asarray([x['token_count'] for x in qw_excerpt_meta])
    y_qwen = np.asarray(qw_cosine_distances)

    x_gemm = np.asarray([x['token_count'] for x in ge_excerpt_meta])
    y_gemm = np.asarray(ge_cosine_distances)

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    ax.scatter(x_qwen,
               y_qwen,
               s=18,
               alpha=0.4,
               marker='o',
               facecolors='none',
               edgecolors='blue',
               label='qwen')

    ax.scatter(x_gemm,
               y_gemm,
               s=22,
               alpha=0.4,
               marker='^',
               facecolors='none',
               edgecolors='orange',
               label='gemma')
    
    ax.set_yscale("log")

    ax.set_xlabel("Number of tokens")
    ax.set_ylabel("Cosine distance")
    ax.legend(frameon=False,
              fontsize=8)

    plt.tight_layout()
    plt.savefig("run12_token_count_vs_cosine_distance_summaries.tiff",
                dpi=600,
                bbox_inches="tight")
    plt.savefig("run12_token_count_vs_cosine_distance_summaries.svg", bbox_inches="tight")
    plt.show()
    plt.close()


if __name__ == '__main__':
    excerpt_and_summary_rank_analysis()
    doc_position_shift()
    '''
    X = np.array([[1.000, 0.000], [0.985, 0.174], [0.940, 0.342], [0.866, 0.500], [0.766, 0.643]])
    
    Y = np.array([[1.00, 0.00], [0.866, 0.500], [0.940, 0.342], [0.985, 0.174], [0.766, 0.643]])

    taus = neighbor_agreement(X, Y)

    for x in taus:
        print(x)
    
    '''
