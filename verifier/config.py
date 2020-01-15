data_folder = "/data"


class Samples:
    # contains subfolders (categories)
    app_metadata = data_folder + "/samples/metadata/"

    samples_database = data_folder + "/samples/samples_database.pk"
    list_of_cross_platform_apps = data_folder + "/samples/cross_platform_apps.txt"

    test_set_save_file = data_folder + "/test_apps.txt"

    # 50% most downloaded apps, 50% least downloaded apps
    # only english
    num_for_test_set = 1000


class Clustering:
    min_downloads_visualize = 4e6


class Embeddings:
    embedding_dim = 301  # +1 for flags
    #downloaded_embedding_file = data_folder + "/word_embeddings/word2vec-wiki-news-300d-1M.vec"
    downloaded_embedding_file = data_folder + "/word_embeddings/glove.6B.300d.txt"
    cached_embedding_file_indices = data_folder + "/word_embeddings/cached-indices.pk"
    cached_embedding_file_values = data_folder + "/word_embeddings/cached-values.np"
    cached_embedding_file_check = data_folder + "/word_embeddings/cache-check.txt"


class TFIDFClassifier:
    
    batch_size = 32
    validation_split = 0.2
    early_stopping_patience = 16
    max_train_epochs = 300

    class ModelMethods:
        enabled = True
        learning_rate = 0.0005
        layers = [5891]
        dropout = 0.38

    class ModelStrings:
        enabled = True
        learning_rate = 0.001
        layers = [2898, 3105]
        dropout = 0.3

    class ModelResourceIds:
        enabled = True
        learning_rate = 0.001
        layers = [2968, 3265, 1393]
        dropout = 0.02


class TrainedModels:
    models_dir = data_folder + "/trained_models/"
    app2text_ids = models_dir + "app2text-ids.h5"
    app2text_stringres = models_dir + "app2text-stringres.h5"
    app2text_methods = models_dir + "app2text-methods.h5"


class TFIDFModels:
    tfidf_folder = data_folder + "/tfidfs/"

    min_terms_per_doc = 5

    description_model = tfidf_folder + "descriptions-model.pk"
    description_data = tfidf_folder + "descriptions-data.pk"

    description_model_2 = tfidf_folder + "descriptions-model-2.pk"
    description_data_2 = tfidf_folder + "descriptions-data-2.pk"

    code_ids_model = tfidf_folder + "code-ids-model.pk"
    code_ids_data = tfidf_folder + "code-ids-data.pk"

    code_stringres_model = tfidf_folder + "code-stringres-model.pk"
    code_stringres_data = tfidf_folder + "code-stringres-data.pk"

    code_methods_model = tfidf_folder + "code-methods-model.pk"
    code_methods_data = tfidf_folder + "code-methods-data.pk"


class Stemming:
    unstem_file = data_folder + '/unstem.pk'
