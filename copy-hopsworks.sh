#!/bin/bash

if [ $# -ne 1 ] ; then
    printf "Please enter project name: "
    read  PROJECT
    echo ""
else
    PROJECT=$1
fi

echo "Copying files from AML repo to Hopsworks."
echo "This will take a little bit..."

hdfs dfs -copyFromLocal -f demodata/*.csv /Projects/${PROJECT}/Resources
zip -r adversarialaml.zip adversarialaml
hdfs dfs -copyFromLocal -f adversarialaml.zip /Projects/${PROJECT}/Resources
hdfs dfs -copyFromLocal -f savedmodels/NodeEmbeddings /Projects/${PROJECT}/Resources
hdfs dfs -copyFromLocal -f savedmodels/ganAml /Projects/${PROJECT}/Resources
hdfs dfs -copyFromLocal -f *.ipynb /Projects/${PROJECT}/Jupyter
hdfs dfs -copyFromLocal -f *.py /Projects/${PROJECT}/Resources
hdfs dfs -copyFromLocal -f *.txt /Projects/${PROJECT}/Resources

echo "Done!"
