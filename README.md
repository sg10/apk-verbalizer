APK Verbalizer
==============

**tl;dr an occurrence-based representation of the app's resource identifiers, string constants, and API method calls, is used to predict phrases of the description.**

The APK verbalizer extracts static parts from an APK as well as source code.
In particular, given an individual application package, including source code and resource files, the system should come up with an overall picture of what the app does. 
Therefore, three different neural networks are trained to output description fragments. The first network receives all resource identifiers an app holds in its XML files. The second one gets string constants found in the app's source code and its XML files. The third model gets all method calls an app makes to the Android API. All inputs and outputs, i.e., descriptions, identifiers, strings, and methods, are represented via term frequency--inverse document frequency (TF-IDF), a commonly used text representation algorithm. After training, the SHAP framework is used to provide explanations for the predictions, e.g., in case a predicted description phrase is *photo frame* it might be inferred from static resources in the app that contain tokens like *photo*, *crop*, *frame*, or similar. Model explanations make the predictions more reliable since we use non-manually labeled samples.
The three models of this description inference system often output very concise, generalizing illustrations of what an app does.



```bash
conda install -c anaconda keras nltk pillow scikit-learn
conda install -c conda-forge langdetect shap
pip install androguard javalang
```