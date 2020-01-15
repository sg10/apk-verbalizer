
def train_summary(eval_values, model, generator):
    #print("epochs: ", len(eval_values['loss']))

    evaluation = list(zip(model.metrics_names, eval_values))
    for e in evaluation:
        print("%20s    %.5f" % (e[0], e[1]))
