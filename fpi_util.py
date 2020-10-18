from app_config import config_manager, AnimalLine, Treatment, Stimulation, Genotype


def categorize(experiment_list, filter = AnimalLine.__name__.lower()):
    """
    Experiments is what?
    Here we take the list of categories from the configuration and depending on the choice of the filter we split
    the experiments to the type of filter they belong using a dictionary.
    The values of the dictionaries are
    :param experiments:
    :param choice:
    :return:
    """
    print(f'categorizing abased on {filter}')
    if filter == AnimalLine.__name__.lower() or filter == '':
        # get the animal filter enum from the configuration
        applied_filter = AnimalLine
    elif filter == Stimulation.__name__.lower():
        applied_filter = Stimulation
    elif filter == Treatment.__name__.lower():
        applied_filter = Treatment
    else:
        print(filter)
        raise ValueError(f'Unknown filter {filter}.')

    # build a dictionary using the animal filter as keys
    # in every animal line we will create a dict that will have as keys the genotypes
    filter_dict = dict.fromkeys([l for l in applied_filter])
    for f in filter_dict:
        filter_dict[f] = dict.fromkeys([gen for gen in Genotype])
        # and then we will create a list as the value of each genotype keyt
        for genotype in filter_dict[f]:
            filter_dict[f][genotype] = []

    # Finally add the experiments based on their attribues (enum values)
    for exp in experiment_list:
        # Get the enum value of the experiment that corresponds to the applied filter. This is the same name with the FPIExperiment attribute.
        f = getattr(exp, applied_filter.__name__.lower())
        print(f)
        filter_dict[getattr(applied_filter, f.name)][exp.genotype].append(exp)
    return filter_dict

def clear_data(genotype_dict):
    for gen_key, filter in genotype_dict.copy().items():
        if not any(genotype_dict[gen_key].values()):
            del genotype_dict[gen_key]
            continue
        else:
            for key, item in filter.copy().items():
                print(f'Item {item}')
                if not any(item):
                    del genotype_dict[gen_key][key]



if __name__ == '__main__':
    print(categorize([1, 2, 3], 'treatment'))
