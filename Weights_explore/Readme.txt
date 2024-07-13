Info:

To execute the firing rate vs weight program

First define the weight interval to explore in Create_arguments, execute it 
python Create_arguments.py

Then, execute the main program

parallel --jobs 3 --bar -a arguments.txt python main.py
