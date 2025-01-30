"""
This module only contains the logo
"""

LOGO = """                                       _
               Powered by              _
       ___  ___                        _
      /  / /  /                     _  _
     /  //  /____  ___  _  __ ___ _(_) _
    /  _   |/ __ |/ _ || |/ / __ '/ /  _
   /  / |  | /_/ / ___| > </ /_/ / /   _
  /__/  |__|____/|___//_/|_|__._/_/    _
                                       _
                                       _"""


def print_logo():
    """
    this function prints it
    """
    for line in LOGO.split("\n"):
        print("    *"+line[:-1]+"*")
