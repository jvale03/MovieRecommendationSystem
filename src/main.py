import ContentBasedFiltering
import CollaborativeFiltering
import sys
from ModelsCombiner import combine_results

def test_input(arg):
    CollaborativeFiltering.read_csvs(False)
    print("\033[32mLoading models...\033[m")
    try:
        content_model = ContentBasedFiltering.load_model()
        collaborative_model = CollaborativeFiltering.load_model()
        print("\033[32mModels loaded!\033[m")
    except Exception as e:
        print(f"\033[31mError: {e}\033[m")
        print("Try to save a Collaborative Filtering model first!\033[m")

    try: 
        content_result = ContentBasedFiltering.content_based_filtering(arg,content_model[0],content_model[1])
        collaborative_result = CollaborativeFiltering.collaborative_filtering(arg,collaborative_model[0],collaborative_model[1],collaborative_model[2])
    
    except Exception as e:
        print(f"\033[31mError: {e}\033[m")

    if len(content_result) or len(collaborative_result):
        print(f"Movie: {arg}\n")
        
        result = combine_results(content_result,collaborative_result)

        for i,movie in enumerate(result):
            print(f"{i+1}: {movie}")

    else:
        print(f"\033[31mNo movies found!\033[m")



def main():
    if len(sys.argv) > 1:
        test_input(sys.argv[1])

    else:
        choice = 0
        while True:
            print("---------------")
            choice = input("1: Process DataSet (if it's the first time)\n2: Process models\n2: Get recomendation\nSelect option: ")
            if choice.isdigit():
                choice = int(choice)
                if choice > 3 or choice < 1:
                    print("\033[31mInvalid!\033[m")
                else:
                        break
            else:
                print("\033[31mInvalid!\033[m")
            
        print("---------------")
        if choice == 1:
            print(f"\033[32mProcessing DataSet...\033[m")
            try:
                ContentBasedFiltering.process_dataset()
                print(f"\033[32mDataSet Processed!\033[m")
            except Exception as e:
                print(f"\033[31mError: {e}!\033[m")
                

        elif choice == 2:
            CollaborativeFiltering.read_csvs(True)
            content_model = ContentBasedFiltering.data_vectorizer()
            collaborative_model = CollaborativeFiltering.algorithm_prepare()
    
            true_false = input(f"Save models? y/n: ")
            if true_false == "y" or true_false == "":
                    print("\033[32mSaving models...\033[m")
                    try:
                        ContentBasedFiltering.save_model(content_model)
                        CollaborativeFiltering.save_model(collaborative_model)
                        print("\033[32mModel saved!\033[m")
                    except Exception as e:
                        print(f"\033[31mError: {e}\033[m")


        elif choice == 3:
            movie_name = input("Insert your movie: ")
            test_input(movie_name)
            

main()