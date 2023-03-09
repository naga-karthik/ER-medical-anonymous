"""
Implements the CL metrics defined in GEM (https://arxiv.org/pdf/1706.08840.pdf) 
and TAG (https://arxiv.org/pdf/2105.05155.pdf) papers
"""

import numpy as np


def BWT(result_matrix):
    """
    Backward Transfer metric

    :param result_matrix: TxT matrix containing accuracy values in each (i, j) entry.
    (i, j) -> test accuracy on task j after training on task i
    """

    # # GEM-version of BWT
    # final_accs = result_matrix[-1, :]  # take accuracies after final training
    # # accuracies on task i right after training on task i, for all i
    # training_accs = np.diag(result_matrix)
    # task_bwt = final_accs - training_accs  # BWT for each task
    # average_bwt = np.mean(task_bwt)  # compute average
    # return average_bwt, task_bwt

    # Mai et al. (https://arxiv.org/pdf/2101.10423.pdf) version of BWT
    avg_bwt, avg_bwtp = 0, 0
    num_domains = result_matrix.shape[0]
    for i in range(1, num_domains):
        for j in range(i-1):
            avg_bwt += (result_matrix[i, j] - result_matrix[j, j])
            avg_bwtp += max((result_matrix[i, j] - result_matrix[j, j]), 0)
    
    avg_bwt /= (num_domains * (num_domains - 1) / 2)
    avg_bwtp /= (num_domains * (num_domains - 1) / 2)
    return avg_bwt, avg_bwtp

def FWT(result_matrix):
    """
    Forward Transfer metric
    :param result_matrix: TxT matrix containing accuracy values in each (i, j) entry.
            (i, j) -> test accuracy on task j after training on task i
    :param single_task_res: 1xT matrix containing single-task accuracies on random init
    """
    # # GEM version of FWT
    # num_domains = result_matrix.shape[0]
    # task_fwt = np.zeros(num_domains)
    # for k in range(1, num_domains):
    #         task_fwt[k] = result_matrix[k - 1, k] - single_task_res[k]
    # avg_fwt = np.mean(task_fwt)
    # return avg_fwt, task_fwt

    # Mai et al. (https://arxiv.org/pdf/2101.10423.pdf) version of FWT
    num_domains = result_matrix.shape[0]
    avg_fwt = np.sum(np.triu(result_matrix, k=1)) / (num_domains * (num_domains - 1) / 2)
    return avg_fwt

def ACC(result_matrix):
    """
    Average Accuracy metric

    :param result_matrix: TxT matrix containing accuracy values in each (i, j) entry.
    (i, j) -> test accuracy on task j after training on task i
    """
    final_accs = result_matrix[-1, :]  # take accuracies after final training
    acc = np.mean(final_accs)  # compute average
    return acc, final_accs

def LA_and_TL(result_matrix):
    """
    Computes the Learning Accuracy and Transfer Learning metrics
    Learning accuracy refers to the performance of the model on a task immediately after 
    training on that task, essentially the diagonal of the result matrix
    Transfer learning (defined in https://arxiv.org/pdf/2005.00079.pdf ) is the average 
    of the learning accuracies (i.e. the diagonal of the result matrix)

    :param result_matrix: TxT matrix containing accuracy values in each (i, j) entry.
    (i, j) -> test accuracy on task j after training on task i
    returns the test accuracy on a task immediately after training on that task
    """ 
    
    learning_accs = np.diag(result_matrix)
    transfer_learning = np.mean(learning_accs)

    return learning_accs, transfer_learning


if __name__ == "__main__":
    # test
    result_matrix = np.array([[0.7, 0.8, 0.7,],
                              [0.85, 0.8, 0.6,],
                              [0.9, 0.85, 0.9,]])
    # single_task_res = np.array([0.1, 0.2, 0.3, 0.4])
    print(BWT(result_matrix))
    print(FWT(result_matrix))
    print(LA_and_TL(result_matrix))
    # print(ACC(result_matrix))