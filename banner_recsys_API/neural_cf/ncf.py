import logging
import pandas as pd
import numpy as np
import random
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import pandas as pd
import math
import heapq
from tqdm import tqdm
from argparse import ArgumentParser


def load_dataset(df):
    """
    Loads dataset into a pandas dataframe
    and transforms it into the format we need.
    We then split it into a training and a test set.
    """


    # Add column names
    df = df.drop(columns = 'Unnamed: 0', axis=1)
    df.columns = ['user', 'item', 'rating']

    # Drop any rows with empty cells or rows
    # with a play count of zero.
    df = df.dropna()
    df = df.loc[df.rating != 0]

    # Remove any users with fewer than 1 interaction.
    df_count = df.groupby(['user']).count()
    df['count'] = df.groupby('user')['user'].transform('count')
    df = df[df['count'] > 1]

    # Convert artists names into numerical IDs
    df['user_id'] = df['user'].astype("category").cat.codes
    df['item_id'] = df['item'].astype("category").cat.codes

    # Create a lookup frame so we can get the artist
    # names back in readable form later.
    item_lookup = df[['item', 'item_id']].drop_duplicates()
    item_lookup['item_id'] = item_lookup.item_id.astype(str)

    user_lookup = df[['user', 'user_id']].drop_duplicates()
    user_lookup['user_id'] = user_lookup.user_id.astype(str)
    user_lookup.to_csv('/home/ubuntu/django-api/neural_cf/user_lookup.csv')
    item_lookup.to_csv('/home/ubuntu/django-api/neural_cf/item_lookup.csv')

    # Grab the columns we need in the order we need them.
    df = df[['user_id', 'item_id', 'rating']]

    # Create training and test sets.
    df_train, df_test = train_test_split(df)
    print(df_train,df_test)

    # Create lists of all unique users and artists
    users = list(np.sort(df.user_id.unique()))
    items = list(np.sort(df.item_id.unique()))

    # Get the rows, columns and values for our matrix.
    rows = df_train.user_id.astype(int)
    cols = df_train.item_id.astype(int)

    values = list(df_train.rating)

    # Get all user ids and item ids.
    uids = np.array(rows.tolist())
    iids = np.array(cols.tolist())

    # Sample 100 negative interactions for each user in our test data
    df_neg = get_negatives(uids, iids, items, df_test)

    return uids, iids, df_train, df_test, df_neg, users, items, item_lookup


def get_negatives(uids, iids, items, df_test):
    """Returns a pandas dataframe of N negative interactions
    based for each user in df_test.
    Args:
        uids (np.array): Numpy array of all user ids.
        iids (np.array): Numpy array of all item ids.
        items (list): List of all unique items.
        df_test (dataframe): Our test set.
    Returns:
        df_neg (dataframe): dataframe with N negative items
            for each (u, i) pair in df_test.
    """

    negativeList = []
    test_u = df_test['user_id'].values.tolist()
    test_i = df_test['item_id'].values.tolist()

    test_ratings = list(zip(test_u, test_i))
    zipped = set(zip(uids, iids))

    for (u, i) in test_ratings:
        negatives = []
        negatives.append((u, i))
        for t in range(100):
            j = np.random.randint(len(items))  # Get random item id.
            while (u, j) in zipped:  # Check if there is an interaction
                j = np.random.randint(len(items))  # If yes, generate a new item id
            negatives.append(j)  # Once a negative interaction is found we add it.
        negativeList.append(negatives)

    df_neg = pd.DataFrame(negativeList)

    return df_neg


def mask_first(x):
    """
    Return a list of 0 for the first item and 1 for all others
    """
    result = np.ones_like(x)
    result[0] = 0

    return result


def train_test_split(df):
    """
    Splits our original data into one test and one
    training set.
    The test set is made up of one item for each user. This is
    our holdout item used to compute Top@K later.
    The training set is the same as our original data but
    without any of the holdout items.
    Args:
        df (dataframe): Our original data
    Returns:
        df_train (dataframe): All of our data except holdout items
        df_test (dataframe): Only our holdout items.
    """

    # Create two copies of our dataframe that we can modify
    df_test = df.copy(deep=True)
    df_train = df.copy(deep=True)

    # Group by user_id and select only the first item for
    # each user (our holdout).
    df_test = df_test.groupby(['user_id']).first()
    df_test['user_id'] = df_test.index
    df_test = df_test[['user_id', 'item_id', 'rating']]
#    del df_test.index.name

    # Remove the same items as we for our test set in our training set.
    mask = df.groupby(['user_id'])['user_id'].transform(mask_first).astype(bool)
    df_train = df.loc[mask]

    return df_train, df_test


def get_train_instances():
    """Samples a number of negative user-item interactions for each
    user-item pair in our testing data.
    Returns:
        user_input (list): A list of all users for each item
        item_input (list): A list of all items for every user,
            both positive and negative interactions.
        labels (list): A list of all labels. 0 or 1.
    """

    user_input, item_input, labels = [], [], []
    zipped = set(zip(uids, iids))

    for (u, i) in zip(uids, iids):
        # Add our positive interaction
        user_input.append(u)
        item_input.append(i)
        labels.append(1)

        # Sample a number of random negative interactions
        for t in range(num_neg):
            j = np.random.randint(len(items))
            while (u, j) in zipped:
                j = np.random.randint(len(items))
            user_input.append(u)
            item_input.append(j)
            labels.append(0)

    return user_input, item_input, labels


def random_mini_batches(U, I, L, mini_batch_size):
    """Returns a list of shuffeled mini batched of a given size.
    Args:
        U (list): All users for every interaction
        I (list): All items for every interaction
        L (list): All labels for every interaction.
    Returns:
        mini_batches (list): A list of minibatches containing sets
            of batch users, batch items and batch labels
            [(u, i, l), (u, i, l) ...]
    """

    mini_batches = []

    shuffled_U = random.sample(U,len(U))
    shuffled_I = random.sample(I,len(I))
    shuffled_L = random.sample(L,len(L))

    #print(shuffled_U,shuffled_I,shuffled_U)

    num_complete_batches = int(math.floor(len(U) / mini_batch_size))
    for k in range(0, num_complete_batches):
        mini_batch_U = shuffled_U[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch_I = shuffled_I[k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch_L = shuffled_L[k * mini_batch_size: k * mini_batch_size + mini_batch_size]

        mini_batch = (mini_batch_U, mini_batch_I, mini_batch_L)
        mini_batches.append(mini_batch)

    if len(U) % mini_batch_size != 0:
        mini_batch_U = shuffled_U[num_complete_batches * mini_batch_size: len(U)]
        mini_batch_I = shuffled_I[num_complete_batches * mini_batch_size: len(U)]
        mini_batch_L = shuffled_L[num_complete_batches * mini_batch_size: len(U)]

        mini_batch = (mini_batch_U, mini_batch_I, mini_batch_L)
        mini_batches.append(mini_batch)

    return mini_batches


def get_hits(k_ranked, holdout):
    """Return 1 if an item exists in a given list and 0 if not. """

    for item in k_ranked:
        if item == holdout:
            return 1
    return 0


def eval_rating(idx, test_ratings, test_negatives, K):
    """Generate ratings for the users in our test set and
    check if our holdout item is among the top K highest scores.
    Args:
        idx (int): Current index
        test_ratings (list): Our test set user-item pairs
        test_negatives (list): 100 negative items for each
            user in our test set.
        K (int): number of top recommendations
    Returns:
        hr (list): A list of 1 if the holdout appeared in our
            top K predicted items. 0 if not.
    """

    map_item_score = {}

    # Get the negative interactions our user.
    items = test_negatives[idx]

    # Get the user idx.
    user_idx = test_ratings[idx][0]

    # Get the item idx, i.e. our holdout item.
    holdout = test_ratings[idx][1]

    # Add the holdout to the end of the negative interactions list.
    items.append(holdout)

    # Prepare our user and item arrays for tensorflow.
    predict_user = np.full(len(items), user_idx, dtype='int32').reshape(-1, 1)
    np_items = np.array(items).reshape(-1, 1)

    # Feed user and items into the TF graph .
    predictions = session.run([output_layer], feed_dict={user: predict_user, item: np_items})

    # Get the predicted scores as a list
    predictions = predictions[0].flatten().tolist()

    # Map predicted score to item id.
    for i in range(len(items)):
        current_item = items[i]
        map_item_score[current_item] = predictions[i]

    # Get the K highest ranked items as a list
    k_ranked = heapq.nlargest(K, map_item_score, key=map_item_score.get)

    # Get a list of hit or no hit.
    hits = get_hits(k_ranked, holdout)

    return hits,k_ranked


def evaluate(df_neg, K=20):
    """Calculate the top@K hit ratio for our recommendations.
    Args:
        df_neg (dataframe): dataframe containing our holdout items
            and 100 randomly sampled negative interactions for each
            (user, item) holdout pair.
        K (int): The 'K' number of ranked predictions we want
            our holdout item to be present in.
    Returns:
        hits (list): list of "hits". 1 if the holdout was present in
            the K highest ranked predictions. 0 if not.
    """

    hits = []

    test_u = df_test['user_id'].values.tolist()
    test_i = df_test['item_id'].values.tolist()

    test_ratings = list(zip(test_u, test_i))
    #print(test_ratings)
    #print(type(test_ratings))
    #test_ratings[idx][1]

    #initialize a dataframe for Top-N products for each user
    topN_df = pd.DataFrame(columns=np.arange(len(test_ratings)))

    df_neg = df_neg.drop(df_neg.columns[0], axis=1)
    test_negatives = df_neg.values.tolist()

    for idx in range(len(test_ratings)):
        # For each idx, call eval_one_rating
        hitrate, topk_ranked = eval_rating(idx, test_ratings, test_negatives, K)
        topN_df[idx] = topk_ranked
        hits.append(hitrate)

    return hits,topN_df

def parse_args():

    parser = ArgumentParser(description='Parse Train value')
    parser.add_argument('--train', type=bool, default=False,
                        help='Input train value.')

    return parser.parse_args()

if __name__ == '__main__':


    data = pd.read_csv('/home/ubuntu/django-api/datasets/ratings_df.csv')
    uids, iids, df_train, df_test, df_neg, users, items, item_lookup = load_dataset(data)

    root = logging.getLogger()
    if root.handlers:
        root.handlers = []
    logging.basicConfig(format='%(asctime)s : %(message)s',
                        filename='neural_cf.log',
                        level=logging.INFO)

    logging.info('Start....')

    # -------------
    # HYPERPARAMS
    # -------------

    num_neg = 6
    latent_features = 8
    epochs = 10
    batch_size = 512
    learning_rate = 0.001

    logging.info('num_neg: {0},latent_features: {1},epochs: {2},'
                 'batch_size: {3},learning_rate: {4}'.format(num_neg,latent_features,epochs,batch_size,learning_rate))

    # -------------------------
    # TENSORFLOW GRAPH
    # -------------------------
    args = parse_args()

    train = args.train
    print(train)
    #train = False

    graph = tf.Graph()

    with graph.as_default():

        # Define input placeholders for user, item and label.
        user = tf.placeholder(tf.int32, shape=(None, 1))
        item = tf.placeholder(tf.int32, shape=(None, 1))
        label = tf.placeholder(tf.int32, shape=(None, 1))

        # User embedding for MLP
        mlp_u_var = tf.Variable(tf.random_normal([len(users), 32], stddev=0.05),
                                    name='mlp_user_embedding')
        mlp_user_embedding = tf.nn.embedding_lookup(mlp_u_var, user)

        # Item embedding for MLP
        mlp_i_var = tf.Variable(tf.random_normal([len(items), 32], stddev=0.05),
                                    name='mlp_item_embedding')
        mlp_item_embedding = tf.nn.embedding_lookup(mlp_i_var, item)

        # User embedding for GMF
        gmf_u_var = tf.Variable(tf.random_normal([len(users), latent_features],
                                                     stddev=0.05), name='gmf_user_embedding')
        gmf_user_embedding = tf.nn.embedding_lookup(gmf_u_var, user)

        # Item embedding for GMF
        gmf_i_var = tf.Variable(tf.random_normal([len(items), latent_features],
                                                     stddev=0.05), name='gmf_item_embedding')
        gmf_item_embedding = tf.nn.embedding_lookup(gmf_i_var, item)

        # Our GMF layers
        gmf_user_embed = tf.keras.layers.Flatten()(gmf_user_embedding)
        gmf_item_embed = tf.keras.layers.Flatten()(gmf_item_embedding)
        gmf_matrix = tf.multiply(gmf_user_embed, gmf_item_embed)

        # Our MLP layers
        mlp_user_embed = tf.keras.layers.Flatten()(mlp_user_embedding)
        mlp_item_embed = tf.keras.layers.Flatten()(mlp_item_embedding)
        mlp_concat = tf.keras.layers.concatenate([mlp_user_embed, mlp_item_embed])

        mlp_dropout = tf.keras.layers.Dropout(0.2)(mlp_concat)

        mlp_layer_1 = tf.keras.layers.Dense(64, activation='relu', name='layer1')(mlp_dropout)
        mlp_batch_norm1 = tf.keras.layers.BatchNormalization(name='batch_norm1')(mlp_layer_1)
        mlp_dropout1 = tf.keras.layers.Dropout(0.2, name='dropout1')(mlp_batch_norm1)

        mlp_layer_2 = tf.keras.layers.Dense(32, activation='relu', name='layer2')(mlp_dropout1)
        mlp_batch_norm2 = tf.keras.layers.BatchNormalization(name='batch_norm1')(mlp_layer_2)
        mlp_dropout2 = tf.keras.layers.Dropout(0.2, name='dropout1')(mlp_batch_norm2)

        mlp_layer_3 = tf.keras.layers.Dense(16, activation='relu', name='layer3')(mlp_dropout2)
        mlp_layer_4 = tf.keras.layers.Dense(8, activation='relu', name='layer4')(mlp_layer_3)

        # We merge the two networks together
        merged_vector = tf.keras.layers.concatenate([gmf_matrix, mlp_layer_4])

        # Our final single neuron output layer.
        output_layer = tf.keras.layers.Dense(1,kernel_initializer="lecun_uniform",name='output_layer')(merged_vector)

        # Our loss function as a binary cross entropy.
        loss = tf.losses.sigmoid_cross_entropy(label, output_layer)

        # Train using the Adam optimizer to minimize our loss.
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        step = opt.minimize(loss)

        # Initialize all tensorflow variables.
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

    session = tf.Session(config=None, graph=graph)
    session.run(init)

    if train is True:


        for epoch in range(epochs):

            # Get our training input.
            user_input, item_input, labels = get_train_instances()

            # Generate a list of minibatches.
            #print(user_input,item_input,labels)
            minibatches = random_mini_batches(user_input, item_input, labels,mini_batch_size=512)

            # This has nothing to do with tensorflow but gives
            # us a nice progress bar for the training
            progress = tqdm(total=len(minibatches))

            # Loop over each batch and feed our users, items and labels
            # into our graph.
            for minibatch in minibatches:
                feed_dict = {user: np.array(minibatch[0]).reshape(-1, 1),
                             item: np.array(minibatch[1]).reshape(-1, 1),
                             label: np.array(minibatch[2]).reshape(-1, 1)}

                # Execute the graph.
                _, l = session.run([step, loss], feed_dict)

                # Update the progress
                progress.update(1)
                progress.set_description('Epoch: %d - Loss: %.3f' % (epoch + 1, l))

            progress.close()

            logging.info('Epoch:{0} - Loss: {1}'.format(epoch+1, l))

            saver.save(session,'./checkpoints/ncf-model',global_step=epoch + 1)

    else:
        test_epoch = 10
        ckpt = tf.train.get_checkpoint_state('./checkpoints')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(session, './checkpoints/ncf-model-{}'.format(test_epoch))


    # Calculate top@K

    hits,topK_products = evaluate(df_neg)
    print('\nTop-K product preferences for each user:\n{}'.format(topK_products))
    print('\n Hit-Ratio(HR):{}'.format(np.array(hits).mean()))
    topK_products.to_csv('/home/ubuntu/django-api/neural_cf/top20products.csv')
