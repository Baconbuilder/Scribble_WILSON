tasks = {}

tasks['voc'] = {
    "offline":
        {
            0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        },
    "19-1":
        {
            0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            1: [20],
        },
    "19-1b":
        {
            0: [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            1: [5],
        },
    "15-5":
        {
            0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            1: [16, 17, 18, 19, 20]
        },
    "15-1":
        {
            0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            1: [16],
            2: [17],
            3: [18],
            4: [19],
            5: [20]
        },
    "10-5":
        {
            0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            1: [11, 12, 13, 14, 15],
            2: [16, 17, 18, 19, 20]
        },
    "10-2":
        {
            0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            1: [11, 12],
            2: [13, 14],
            3: [15, 16],
            4: [17, 18],
            5: [19, 20]
        },
    "10-10":
        {
            0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            1: [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        }
}


def get_task_list():
    return [task for ds in tasks.keys() for task in tasks[ds].keys()]


def get_task_labels(dataset, name, step):
    if dataset in tasks and name in tasks[dataset]:
        task_dict = tasks[dataset][name]
    else:
        raise NotImplementedError
    assert step in task_dict.keys(), f"You should provide a valid step! [{step} is out of range]"

    labels = list(task_dict[step])
    labels_old = [label for s in range(step) for label in task_dict[s]]
    return labels, labels_old, f'{dataset}/{name}'


def get_task_dict(dataset, name, step):
    if dataset in tasks and name in tasks[dataset]:
        task_dict = tasks[dataset][name]
    else:
        raise NotImplementedError
    assert step in task_dict.keys(), f"You should provide a valid step! [{step} is out of range]"

    class_map = {s: task_dict[s] for s in range(step+1)}
    return class_map


def get_per_task_classes(dataset, name, step):
    if dataset in tasks and name in tasks[dataset]:
        task_dict = tasks[dataset][name]
    else:
        raise NotImplementedError
    assert step in task_dict.keys(), f"You should provide a valid step! [{step} is out of range]"

    classes = [len(task_dict[s]) for s in range(step+1)]
    return classes
