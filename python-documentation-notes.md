# PYTHON DOCUMENTATION GUIDE (NOTES)
## IMPORT STATEMENTS
Imports should always be written at the top of the file, after any module comments and docstrings.<br>
Imports should be divided according to what is being imported. There are generally three groups:
1. Standard library imports (Python’s built-in modules)
2. Related third party imports (modules that are installed and do not belong to the current application)
3. Local application imports (modules that belong to the current application)

Each group of imports should be separated by a blank space.


- Use import statements for packages and modules only, not for individual classes or functions<br>
- Use import x for importing packages and modules.<br>
- Use from x import y where x is the package prefix and y is the module name with no prefix.<br>
- Use from x import y as z if two modules named y are to be imported or if y is an inconveniently long name.<br>
- Use import y as z only when z is a standard abbreviation (e.g., np for numpy).<br>
- Use paranthesis for importing multiple modules from a package
>```from keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense)```

- Use absolute imports whenever possible


### Relative Imports

A single dot means that the module or package referenced is in the same directory as the current location. Two dots mean that it is in the parent directory of the current location—that is, the directory above. Three dots mean that it is in the grandparent directory, and so on. This will probably be familiar to you if you use a Unix-like operating system!


```package/
    __init__.py
    subpackage1/
        __init__.py
        moduleX.py
        moduleY.py
    subpackage2/
        __init__.py
        moduleZ.py
    moduleA.py
```
Assuming that the current file is moduleX.py, following are correct usages of the new syntax:
```
from .moduleY import spam
from .moduleY import spam as ham
from . import moduleY
from ..subpackage1 import moduleY
from ..subpackage2.moduleZ import eggs
from ..moduleA import foo
from ...package import bar
from ...sys import path
```
Note that while that last case is legal, it is certainly discouraged ("insane" was the word Guido used).


## NAMING CONVENTION

<table class="table table-hover">
<thead>
<tr>
<th>Type</th>
<th>Naming Convention</th>
<th>Examples</th>
</tr>
</thead>
<tbody>
<tr>
<td>Function</td>
<td>Use a lowercase word or words. Separate words by underscores to improve readability.</td>
<td><code>function</code>, <code>my_function</code></td>
</tr>
<tr>
<td>Variable</td>
<td>Use a lowercase single letter, word, or words. Separate words with underscores to improve readability.</td>
<td><code>x</code>, <code>var</code>, <code>my_variable</code></td>
</tr>
<tr>
<td>Class</td>
<td>Start each word with a capital letter. Do not separate words with underscores. This style is called camel case.</td>
<td><code>Model</code>, <code>MyClass</code></td>
</tr>
<tr>
<td>Method</td>
<td>Use a lowercase word or words. Separate words with underscores to improve readability.</td>
<td><code>class_method</code>, <code>method</code></td>
</tr>
<tr>
<td>Constant</td>
<td>Use an uppercase single letter, word, or words. Separate words with underscores to improve readability.</td>
<td><code>CONSTANT</code>, <code>MY_CONSTANT</code>, <code>MY_LONG_CONSTANT</code></td>
</tr>
<tr>
<td>Module</td>
<td>Use a short, lowercase word or words. Separate words with underscores to improve readability.</td>
<td><code>module.py</code>, <code>my_module.py</code></td>
</tr>
<tr>
<td>Package</td>
<td>Use a short, lowercase word or words. Do not separate words with underscores.</td>
<td><code>package</code>, <code>mypackage</code></td>
</tr>
</tbody>
</table>

## INDENTATION
- Use 4 consecutive spaces to indicate indentation
- Prefer spaces over tabs

## COMMENTS
- ### Block Comments
    - Indent block comments to the same level as the code they describe.
    - Start each line with a # followed by a single space.
    - Separate paragraphs by a line containing a single #.

- ### Inline Comments
    - Use them sparingly

## DOCUMENTATION STRINGS
The most important rules applying to docstrings are the following:

- Surround docstrings with three double quotes on either side, as in

        """ This is a docstring """
- Write them for all public modules, functions, classes, and methods.
- Put the """ that ends a multiline docstring on a line by itself

Their purpose is to provide your users with a brief overview of the object. At a bare minimum, a docstring should be a quick summary of whatever is it you’re describing and should be contained within a single line.<br>
Multi-lined docstrings are used to further elaborate on the object beyond the summary. All multi-lined docstrings have the following parts:

- A one-line summary line
- A blank line proceeding the summary
- Any further elaboration for the docstring
- Another blank line

### Docstrings can be further broken up into three major categories:

- **Class Docstrings:** Class and class methods
- **Package and Module Docstrings:** Package, modules, and functions
- **Script Docstrings:** Script and functions

1. ### **Class Docstring:**
    Class docstrings should contain the following information:

    - A brief summary of its purpose and behavior
    - Any public methods, along with a brief description
    - Any class properties (attributes)
    - Anything related to the interface for subclassers, if the class is intended to be subclassed

    The class constructor parameters should be documented within the __ init __ class method docstring. Individual methods should be documented using their individual docstrings. Class method docstrings should contain the following:
    - A brief description of what the method is and what it’s used for
    - Any arguments (both required and optional) that are passed including keyword arguments
    - Label any arguments that are considered optional or have a default value
    - Any side effects that occur when executing the method
    - Any exceptions that are raised
    - Any restrictions on when the method can be called


2. ### **Package and Module Docstrings:**

    Package docstrings should be placed at the top of the package’s __ init__.py file. This docstring should list the modules and sub-packages that are exported by the package.

    Module docstrings are similar to class docstrings. Instead of classes and class methods being documented, it’s now the module and any functions found within. Module docstrings are placed at the top of the file even before any imports. Module docstrings should include the following:

    - A brief description of the module and its purpose
    - A list of any classes, exception, functions, and any other objects exported by the module

    The docstring for a module function should include the same items as a class method:

    - A brief description of what the function is and what it’s used for
    - Any arguments (both required and optional) that are passed including keyword arguments
    - Label any arguments that are considered optional
    - Any side effects that occur when executing the function
    - Any exceptions that are raised
    - Any restrictions on when the function can be called

I prefer Google's docstrings format

Ex:
```
"""Gets and prints the spreadsheet's header columns

Parameters:
    file_loc (str): The file location of the spreadsheet
    print_cols (bool): A flag used to print the columns to the console
        (default is False)

Returns:
    list: a list of strings representing the header columns
""
```


