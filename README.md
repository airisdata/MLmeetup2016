# Problem description

The description of the problem can be found here: https://www.kaggle.com/c/dato-native

The task is to predict which web pages served by StumbleUpon company are sponsored.

When native advertising is done right, users aren't desperately scanning an ad for a hidden "x". In fact, they don't even know they're viewing one. To pull this off, native ads need to be just as interesting, fun, and informative as the unpaid content on a site.

If media companies can better identify poorly designed native ads, they can keep them off your feed and out of your user experience. 


# Getting the data

The competition is over, but we can still download the trianing and testing data here: https://www.kaggle.com/c/dato-native/data

You would have to register at Kaggle.com to download the data! It is just one click away.

Note, you would need only need labeled data, which we are going to use for training and cross validation:

##### train\_v2.csv
##### {0,1,2,3,4}.zip

## File descriptions

{0,1,2,3,4}.zip - are all HTML files. 
Files listed in train\_v2.csv are training files:

```bash
file - the raw file name
sponsored - 0: organic content; 1: sponsored content label
```

We are going to use a fraction of fake data for exploratory analysis with Spark in the iPython notebook. Test with the full dataset will be illustrated during the live demo.


# Prerequisites (should preferrably be done before the day of meetup)

To be able to follow the hands-on portion of the meetup, you are going to need a laptop with Anaconda and PySpark installed. We are going to use a fraction of data for that.

Apache Spark is a distributed computing framework, so to get most out of it you would need a cluster. I am going to use an SGI Hadoop Linux cluster consisting of 6 data nodes and 4 service nodes all with Intel Xeon CPU E5-2680 v2 @ 2.80GHz, and each CPU processor core on a worker node having 256 GB of memory for the live demo on the full dataset.
Cloudera distribution of Hadoop with Spark via YARN is going to be used.  


## Install Apache Spark on your laptop

Install Apache Spark 1.6.1 or later on your laptop. Here are some instructions on how to do that: 

1) Go to the Spark download page: http://spark.apache.org/downloads.html select a prebuilt distribution you need. Download and unpack it, then proceed to the step 2).

Alternatively, install and build a custom Spark distribution from source:

```bash
git clone git://github.com/apache/spark.git
cd spark/
build/mvn -DskipTests clean package
```
Note, that we are not building it against Hadoop or YARN.

You are going to need to install Maven build tool for that. You can download Maven from the web: https://maven.apache.org/download.cgi and add to PATH:

```bash
export PATH=<path to your maven>/apache-maven-3.3.9/bin:$PATH
```

2) Update following environmental variables to point to the new Spark location:

```bash
export SPARK_HOME=/home/<your_username>/your_local_spark/spark
export PATH=$SPARK_HOME/sbin:$SPARK_HOME/bin:$PATH
```

these 3 lines should be added to the .profile file on your laptop, otherwise you would have to export these values each time you log in!

## Install Anaconda and pip

To install Anaconda, please go to this page: https://www.continuum.io/downloads and select the version for Python 2.7 compatible with your operating system.

Type:

```bash
conda --help
```

to confirm the installation is succesful.

## Create isolated Anaconda environment for the Meetup

Once this is installed, create an isolated Anaconda nevironment for the meetup, and install ipython notebook in it along side with other dependencies:

```bash
conda create --name meetup_env --file conda-requirements.txt
source activate meetup_env
```
