# Leveraging Corpus Metadata to Detect Template-based Translation: An Exploratory Case Study of the Egyptian Arabic Wikipedia Edition

We, in this repository, share our labeled datasets, extracted corpora, code and scripts of the exploratory analysis, the multivariate machine learning classifiers and clusters, and the implementation and deployment of the best-performing classifier as a web-based detection system called "*Egyptian Arabic Wikipedia Scanner*", which all are introduced in our accepted paper, [**Leveraging Corpus Metadata to Detect Template-based Translation: An Exploratory Case Study of the Egyptian Arabic Wikipedia Edition**](https://arxiv.org/abs/2404.00565), at [*The 6th Workshop on Open-Source Arabic Corpora and Processing Tools (OSACT6)*](https://osact-lrec.github.io/), co-located with [LREC-COLING 2024](https://lrec-coling-2024.org/), 20-25 May 2024. 
### Table of Content

* [**Exploratory Analysis:**](https://github.com/SaiedAlshahrani/leveraging-corpus-metadata/tree/main/Exploratory-Analysis)
	1. [Shallow Content](https://github.com/SaiedAlshahrani/leveraging-corpus-metadata/blob/main/Exploratory-Analysis/Shallow_Content.ipynb)
	2. [Poor Quality Content](https://github.com/SaiedAlshahrani/leveraging-corpus-metadata/blob/main/Exploratory-Analysis/Poor_Content_Quality.ipynb)
	3. [Misleading Human Involvement](https://github.com/SaiedAlshahrani/leveraging-corpus-metadata/blob/main/Exploratory-Analysis/Misleading_Human_Involvement.ipynb)
* [**Experimental Setups:**](https://github.com/SaiedAlshahrani/leveraging-corpus-metadata/tree/main/Experimental-Setups)
	* [Dataset Filtering, Labeling, and Cleaning](https://github.com/SaiedAlshahrani/leveraging-corpus-metadata/blob/main/Experimental-Setups/Dataset_Filtering_Labeling_Cleaning.ipynb)
	* Dataset Encoding Using Spark-NLP & CAMeLBERT:
		- [Encoding with Spark-NLP (Egyptian Word2Vec-CBOW 300D)](https://github.com/SaiedAlshahrani/leveraging-corpus-metadata/blob/main/Experimental-Setups/Dataset_Encoding_Spark-NLP.ipynb)
		- [Encoding with CAMeLBERT (CAMeLBERT-Mix POS-EGY Model)](https://github.com/SaiedAlshahrani/leveraging-corpus-metadata/blob/main/Experimental-Setups/Dataset_Encoding_CAMeLBERT.ipynb)
* [**Template Translation Detection:**](https://github.com/SaiedAlshahrani/leveraging-corpus-metadata/tree/main/Template-Translation-Detection)
	* [Supervised Classification Algorithms:](https://github.com/SaiedAlshahrani/leveraging-corpus-metadata/tree/main/Template-Translation-Detection/Supervised-Classification-Algorithms)
		1. [XGBoost](https://github.com/SaiedAlshahrani/leveraging-corpus-metadata/blob/main/Template-Translation-Detection/Supervised-Classification-Algorithms/XGBoost.ipynb)
		2. [Naive Bayes](https://github.com/SaiedAlshahrani/leveraging-corpus-metadata/blob/main/Template-Translation-Detection/Supervised-Classification-Algorithms/NaiveBayes.ipynb)
		3. [Random Forest](https://github.com/SaiedAlshahrani/leveraging-corpus-metadata/blob/main/Template-Translation-Detection/Supervised-Classification-Algorithms/RandomForest.ipynb)
		4. [Logistic Regression](https://github.com/SaiedAlshahrani/leveraging-corpus-metadata/blob/main/Template-Translation-Detection/Supervised-Classification-Algorithms/LogisticRegression.ipynb)
		5. [Support Vector Machine](https://github.com/SaiedAlshahrani/leveraging-corpus-metadata/blob/main/Template-Translation-Detection/Supervised-Classification-Algorithms/SupportVectorMachine.ipynb)
	* [Unsupervised Clustering Algorithms:](https://github.com/SaiedAlshahrani/leveraging-corpus-metadata/tree/main/Template-Translation-Detection/Unsupervised-Clustering-Algorithms)
		1. [DBSCAN](https://github.com/SaiedAlshahrani/leveraging-corpus-metadata/blob/main/Template-Translation-Detection/Unsupervised-Clustering-Algorithms/DBSCAN.ipynb)
		2. [K-Means](https://github.com/SaiedAlshahrani/leveraging-corpus-metadata/blob/main/Template-Translation-Detection/Unsupervised-Clustering-Algorithms/K-Means.ipynb)
		3. [Hierarchical](https://github.com/SaiedAlshahrani/leveraging-corpus-metadata/blob/main/Template-Translation-Detection/Unsupervised-Clustering-Algorithms/Hierarchical.ipynb)
* Web-based Detection System/Application:
	1. [Best-performing Classifier, XGBoost](https://github.com/SaiedAlshahrani/leveraging-corpus-metadata/blob/main/Template-Translation-Detection/Supervised-Classification-Algorithms/Best-performing-Classifier.ipynb)
	2. [Egyptian Arabic Wikipedia Scanner:](https://github.com/SaiedAlshahrani/Egyptian-Wikipedia-Scanner)
		* [Streamlit Community Cloud](https://egyptian-wikipedia-scanner.streamlit.app/)
		* [Hugging Face Spaces](https://huggingface.co/spaces/SaiedAlshahrani/Egyptian-Wikipedia-Scanner)
* **Corpora and Datasets:**
	* [Arabic Wikipedia Corpora:](https://github.com/SaiedAlshahrani/leveraging-corpus-metadata/tree/main/Wikipedia-20240101)
		1. [Arabic Wikipedia Articles](https://github.com/SaiedAlshahrani/leveraging-corpus-metadata/blob/main/Wikipedia-20240101/arwiki-20240101.zip)
		2. [Egyptian Wikipedia Articles](https://github.com/SaiedAlshahrani/leveraging-corpus-metadata/blob/main/Wikipedia-20240101/arzwiki-20240101.zip)
		3. [Moroccan Wikipedia Articles](https://github.com/SaiedAlshahrani/leveraging-corpus-metadata/blob/main/Wikipedia-20240101/arywiki-20240101.zip)
	* Egyptian *Template-translated* Articles:
		1. [Hugging Face Datasets](https://huggingface.co/datasets/SaiedAlshahrani/Detect-Egyptian-Wikipedia-Articles)
		2. [Raw CSV Datasets](https://github.com/SaiedAlshahrani/leveraging-corpus-metadata/tree/main/Experimental-Setups/csvs)
		3. [Masader+ (#652)](https://arbml.github.io/masader/card?id=652)
		
* **Paper Citations:**

>Saied Alshahrani, Hesham Haroon, Ali Elfilali, Mariama Njie, and Jeanna Matthews. 2024. [Leveraging Corpus Metadata to Detect Template-based Translation: An Exploratory Case Study of the Egyptian Arabic Wikipedia Edition](https://arxiv.org/abs/2404.00565). *arXiv preprint arXiv:2404.00565*.

Saied Alshahrani, Hesham Haroon, Ali Elfilali, Mariama Njie, and Jeanna Matthews. 2024. [Leveraging Corpus Metadata to Detect Template-based Translation: An Exploratory Case Study of the Egyptian Arabic Wikipedia Edition](https://aclanthology.org/2024.osact-1.4/). *In Proceedings of the 6th Workshop on Open-Source Arabic Corpora and Processing Tools (OSACT) with Shared Tasks on Arabic LLMs Hallucination and Dialect to MSA Machine Translation @ LREC-COLING 2024*, pages 31–45, Torino, Italia. ELRA and ICCL.*

```
@article{alshahrani2024leveraging,
      title={Leveraging Corpus Metadata to Detect Template-based Translation: An Exploratory Case Study of the Egyptian Arabic Wikipedia Edition}, 
      author={Saied Alshahrani and Hesham Haroon and Ali Elfilali and Mariama Njie and Jeanna Matthews},
      year={2024},
      eprint={2404.00565},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
      journal={arXiv preprint arXiv:2404.00565},
      url={https://arxiv.org/abs/2404.00565}
}
```
```
@inproceedings{alshahrani-etal-2024-leveraging,
    title = "Leveraging Corpus Metadata to Detect Template-based Translation: An Exploratory Case Study of the {E}gyptian {A}rabic {W}ikipedia Edition",
    author = "Alshahrani, Saied  and
      Mohammed, Hesham Haroon  and
      Elfilali, Ali  and
      Njie, Mariama  and
      Matthews, Jeanna",
    editor = "Al-Khalifa, Hend  and
      Darwish, Kareem  and
      Mubarak, Hamdy  and
      Ali, Mona  and
      Elsayed, Tamer",
    booktitle = "Proceedings of the 6th Workshop on Open-Source Arabic Corpora and Processing Tools (OSACT) with Shared Tasks on Arabic LLMs Hallucination and Dialect to MSA Machine Translation @ LREC-COLING 2024",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.osact-1.4",
    pages = "31--45",
    abstract = "Wikipedia articles (content pages) are commonly used corpora in Natural Language Processing (NLP) research, especially in low-resource languages other than English. Yet, a few research studies have studied the three Arabic Wikipedia editions, Arabic Wikipedia (AR), Egyptian Arabic Wikipedia (ARZ), and Moroccan Arabic Wikipedia (ARY), and documented issues in the Egyptian Arabic Wikipedia edition regarding the massive automatic creation of its articles using template-based translation from English to Arabic without human involvement, overwhelming the Egyptian Arabic Wikipedia with articles that do not only have low-quality content but also with articles that do not represent the Egyptian people, their culture, and their dialect. In this paper, we aim to mitigate the problem of template translation that occurred in the Egyptian Arabic Wikipedia by identifying these template-translated articles and their characteristics through exploratory analysis and building automatic detection systems. We first explore the content of the three Arabic Wikipedia editions in terms of density, quality, and human contributions and utilize the resulting insights to build multivariate machine learning classifiers leveraging articles{'} metadata to detect the template-translated articles automatically. We then publicly deploy and host the best-performing classifier as an online application called {`}Egyptian Wikipedia Scanner{'} and release the extracted, filtered, labeled, and preprocessed datasets to the research community to benefit from our datasets and the online, web-based detection system.",
}
```
