logo = """                                       
               Powered by              
       ___  ___                        
      /  / /  /                     _  
     /  //  /____  ___  _  __ ___ _(_) 
    /  _   |/ __ |/ _ || |/ / __ '/ /  
   /  / |  | /_/ / ___| > </ /_/ / /   
  /__/  |__|____/|___//_/|_|__._/_/    
                                       
                                       """

def print_logo():
    for l in logo.split("\n"):
        print("    *"+l+"*")