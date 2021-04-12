# Names-nonnames-classifier
The objective of this project is to classify English words as names/non-names. We have made use of Character-level
Recurrent Neural Networks to model our data. The idea behind this has been inspired by the prediction of the Domain Generation Algorithms using LSTM.

Data Description:
We used multiple sources like Government census websites, Github, kaggle [8-15] to find out a list
of English names and non-names. We tried to avoid very common words, we
filtered out the words so that they have a minimum length of 4 characters for both names and
non-names. The total number of instances in the English dataset is shown below:

English Dataset:

	Total number of instances: 568825
	Total number of names: 234507
	Total number of non-names: 334318
	
Implementation Approach:
In this project we have performed Names Detection by Training our own character level embeddings where we have allowed an indicator vector to be transformed to a
character level embedding using the (Keras) Embedding layer and then finally pass it to the LSTM layer. In this approach, each lowercase alphabet is mapped to its corresponding number from (1,27) For example: { ‘a’ : 1, ‘b’ : 2, ‘c’ : 3, ... , ‘z’ : 27 } , the vocab gets encoded into vectors. All the words (names and non-names) are padded to maintain a maxlen of 31. Note that we choose 31 because that is the length of the largest word in our dataset. We pass an array of (N*31) as X and the labels as y, where N is the number of training instances. We used the 80:20 ratio to split ourdata into training and testing. In this approach the embeddings and the weights of the networkare trained together.

Model:

	def build_model(max_features, maxlen):
    		"""Build Bi-LSTM model"""
    		model = tf.keras.models.Sequential()
    		model.add(tf.keras.layers.Embedding(max_features, 300, weights=[embedding_matrix], input_length=maxlen, trainable=True))
    		model.add(tf.keras.layers.Bidirectional(tf.compat.v1.keras.layers.CuDNNLSTM(300)))
    		model.add(tf.keras.layers.Dense(300))
    		model.add(tf.keras.layers.Dropout(0.5))
    		model.add(tf.keras.layers.Dense(1))
    		model.add(tf.keras.layers.Activation('sigmoid'))
    		model.compile(loss='binary_crossentropy', optimizer='adam')
    		return model

Results:
Accuracy:

	Class 0: 0.9073260677290239
	Class 1: 0.9215861277572358
	Overall: 0.913224629719158

Classification Report:

               precision    recall  f1-score   support

            0       0.94      0.91      0.92     66707
            1       0.88      0.92      0.90     47058

  	  accuracy                           0.91    113765
   
	 macro avg       0.91      0.91      0.91    113765

	weighted avg     0.91      0.91      0.91    113765






