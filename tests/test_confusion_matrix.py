import numpy as np
from vision.confusion_matrix import generate_confusion_matrix


def test_generate_confusion_matrix():
    """Tests confusion matrix generation on known inputs"""
    ground_truth = np.array([2, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0])
    predicted = np.array([2, 1, 0, 2, 0, 1, 2, 0, 0, 1, 0, 2])

    # fmt: off
    ground_truth_confusion_matrix = np.array([[1, 1, 1],
                                              [2, 1, 1],
                                              [2, 1, 2]])
    # fmt: on
    student_confusion_matrix = generate_confusion_matrix(
        ground_truth, predicted, num_classes=3, normalize=False
    )

    assert np.allclose(
        ground_truth_confusion_matrix, student_confusion_matrix, atol=1e-2
    ), "Confusion matrix is incorrect"


def test_generate_confusion_matrix_normalized():
    """Tests normalized confusion matrix generation on known inputs"""
    ground_truth = np.array([2, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0])
    predicted = np.array([2, 1, 0, 2, 0, 1, 2, 0, 0, 1, 0, 2])

    # fmt: off
    ground_truth_confusion_matrix = np.array([[1/3, 1/3, 1/3],
                                              [1/2, 1/4, 1/4],
                                              [2/5, 1/5, 2/5]])
    # fmt: on

    student_confusion_matrix = generate_confusion_matrix(
        ground_truth, predicted, num_classes=3, normalize=True
    )

    assert np.allclose(
        ground_truth_confusion_matrix, student_confusion_matrix, atol=1e-2
    ), "Normalized confusion matrix is incorrect"
