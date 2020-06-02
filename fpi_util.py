from app_config import config_manager

def categorize(experiment_list, filter = 'AnimalLines'):
    """
    Experiments is what?
    Here we take the list of categories from the configuration and depending on the choice of the filter we split
    the experiments to the type of filter they belong using a dictionary.
    The values of the dictionaries are
    :param experiments:
    :param choice:
    :return:
    """
    genotypes = config_manager.genotypes
    if filter == 'mouselines'.lower() or filter == '':
        # get the animal lines from the configuration
        lines = config_manager.animal_lines
        # build a dictionary using the animal lines as keys
        # in every animal line we will create a dict that will have as keys the genotypes
        c = dict.fromkeys([l.lower() for l in lines])
        for line in c:
            c[line] = dict.fromkeys([gen.lower() for gen in genotypes])
            # and then we will create a list as the value of each genotype keyt
            for genotype in c[line]:
                c[line][genotype] = []
        for exp in experiment_list:
            c[exp.animal_line.lower()][exp.genotype.lower()].append(exp)
        return c
    elif filter == 'stimulations'.lower():
        stim = config_manager.stimulations
        c = dict.fromkeys([s.lower() for s in stim])
        for stim in c:
            c[stim] = dict.fromkeys([gen.lower() for gen in genotypes])
            for genotype in c[stim]:
                c[stim][genotype] = []
        for exp in experiment_list:
            c[exp.stimulation.lower()][exp.genotype.lower()].append(exp)
        return c
    elif filter == 'treatment'.lower():
        treatments = config_manager.treatments
        c = dict.fromkeys([t.lower() for t in treatments])
        for treatment in c:
            c[treatment] = dict.fromkeys([gen.lower() for gen in genotypes])
            for genotype in c[treatment]:
                c[treatment][genotype] = []
        for exp in experiment_list:
            c[exp.treatment.lower()][exp.genotype.lower()].append(exp)
        return c


def genotype_split(exp_list):
    genotypes = config_manager.genotypes
    geno_dict = dict.fromkeys(genotypes)
    for item in geno_dict:
        geno_dict[item] = []
    for exp in exp_list:
        geno_dict[exp.genotype].append(exp)

    return geno_dict