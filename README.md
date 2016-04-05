This is the implementation of word2phrase (see section 4 Learning Phrases of http://arxiv.org/pdf/1310.4546.pdf).

The estimator will take a dataframe as a training set, and produce a model that can be used with the transformer pipeline.

Below is an example using the testSentences.scala file:

First we create our training dataset; it's a dataframe where the occurrences "new york" and "test drive" appears frequently.  (The sentences make no sense as they are randomly generated words.  See below for the full dataframe.)

You can copy/paste this (be sure to include the full dataframe included in the repo) into your spark shell to test it.

First run spark-shell and download the word2phrase mvn package:

  spark-shell --packages com.reputation.spark.word2phrase.1.0.1

Import the algorithm and create the dataframe:

  import org.apache.spark.ml.feature.Word2Phrase

  val wordDataFrame = sqlContext.createDataFrame(Seq(
  (0, "new york test drive cool york how always learn media new york ."),
  (1, "online york new york learn to media cool time ."),
  (2, "media play how cool times play ."),
  (3, "code to to code york to loaded times media ."),
  (4, "play awesome to york ."),
  .
  .
  .
  (1099, "work please ideone how awesome times ."),
  (1100, "play how play awesome to new york york awesome use new york work please loaded always like ."),
  (1101, "learn like I media online new york ."),
  (1102, "media follow learn code code there to york times ."),
  (1103, "cool use play work please york cool new york how follow ."),
  (1104, "awesome how loaded media use us cool new york online code judge ideone like ."),
  (1105, "judge media times time ideone new york new york time us fun ."),
  (1106, "new york to time there media time fun there new like media time time ."),
  (1107, "awesome to new times learn cool code play how to work please to learn to ."),
  (1108, "there work please online new york how to play play judge how always work please ."),
  (1109, "fun ideone to play loaded like how ."),
  (1110, "fun york test drive awesome play times ideone new us media like follow .")
  )).toDF("label", "inputWords")

We set the input and output column names and create the model (the estimator step, represented by the fit(wordDataFrame) function).

  val t = new Word2Phrase().setInputCol("inputWords").setOutputCol("out")

  val model = t.fit(wordDataFrame)

We then use this model to transform our original dataframe sentences and view the results.  Unfortunately you can't see the entire row in the spark-shell, but in the out column it's clear that all instances of "new york" and "test drive" have been transformed into "new_york" and "test_drive".

  val bi_gram_data = model.transform(wordDataFrame)

  bi_gram_data.show()
  //this is the final result
  +-----+--------------------+--------------------+
  |label|          inputWords|                 out|
  +-----+--------------------+--------------------+
  |    0|new york test dri...|new_york test_dri...|
  |    1|online york new y...|online york new_y...|
  |    2|media play how co...|media play how co...|
  |    3|code to to code y...|code to to code y...|
  |    4|play awesome to y...|play awesome to y...|
  |    5|   like I I always .|   like I I always .|
  |    6|how to there lear...|how to there lear...|
  |    7|judge time us pla...|judge time us pla...|
  |    8|judge test drive ...|judge test_drive ...|
  |    9|judge follow fun ...|judge follow fun ...|
  |   10|how I follow ideo...|how I follow ideo...|
  |   11|use use learn I t...|use use learn I t...|
  |   12|us new york alway...|us new_york alway...|
  |   13|there always how ...|there always how ...|
  |   14|always time media...|always time media...|
  |   15|how test drive to...|how test_drive to...|
  |   16|cool us online ti...|cool us online ti...|
  |   17|follow time aweso...|follow time aweso...|
  |   18|us york test driv...|us york test_driv...|
  |   19|use fun new york ...|use fun new_york ...|
  +-----+--------------------+--------------------+
  only showing top 20 rows

See the blog post for more details on word2phrase:

tech.reputation.com