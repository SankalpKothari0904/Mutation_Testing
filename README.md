# Mutation Testing
Results of Mutation Testing on Dynamic Programming Algorithms

## Project Structure

The DP algorithms have been written in 2 different programming languages - Java and Python along with their unit tests.

The unit tests for Java have been written using JUnit and Mutation testing has been carried out using PITest.

For Python, unit testing has been carried out using PyTest and mutation testing using MutPy.

### Installing Python libraries

```
$ pip install pytest
$ pip install mutpy
```

## Running the tests

The Java section of the code has been built and structured as a Maven project.

To see the output of the mutated test cases, simply clone the repository and navigate to the `/Java/ClassicalDP` directory and execute the following commands.

```
$ mvn clean
$ mvn clean test
$ mvn org.pitest:pitest-maven:mutationCoverage
```

The first 2 commands will build the project and run the unit tests. THe third command runs the mutation tests.

The output of the mutation tests can be viewed when user navigates to the `/Java/ClassicalDP/target/pit-reports` directory and opens the `index.html` file in their browser.

<hr>

For running python test cases, navigate to the top level directory of the repo, and run then run the following commands -

```
cd Python 
pytest tests/CLassicalDPTest.py
mut.py --target=ClassicalDP --unit-test=tests.ClassicalDPTest --report-html mutation_report
```

This will run all the unit tests and mutation tests for all the Python functions as well.

The output for the mutation tests can be found in the `mutation_report` directory in the `index.html` file. The sub-directory generated contains each mutation applied along with its result.