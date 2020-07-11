var tag = document.getElementsByTagName("p")[0];
text = tag.innerHTML;
// Here I would like to call the Python interpreter with Python function
arrOfStrings = openSomehowPythonInterpreter("~/pythoncode.py", "processParagraph(text)");