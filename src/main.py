import ContentBasedFiltering
import CollaborativeFiltering
import sys

def test_input(arg):
    CollaborativeFiltering.read_csvs(False)
    print("\033[32mLoading models...\033[m")
    try:
        content_model = ContentBasedFiltering.load_model()
        collaborative_model = CollaborativeFiltering.load_model()
        print("\033[32mModels loaded!\033[m")
    except Exception as e:
        print(f"\033[31mError: {e}")
        print("Try to save a Collaborative Filtering model first!\033[m")

    indices = content_model[0]
    movie_idx = indices[arg]

    print(ContentBasedFiltering.content_based_filtering(movie_idx,content_model[0],content_model[1]))
    print(CollaborativeFiltering.collaborative_filtering(movie_idx,collaborative_model[0],collaborative_model[1],collaborative_model[2]))


def main():
    print("cheguei")
    if len(sys.argv) > 1:
        test_input(sys.argv[1])

    else:
        choice = 0
        while True:
            print("---------------")
            choice = input("1: Process models\n2: Get recomendation\nSelect option: ")
            if choice.isdigit():
                choice = int(choice)
                if choice > 2 or choice < 1:
                    print("\033[31mInvalid!\033[m")
                else:
                        break
            else:
                print("\033[31mInvalid!\033[m")
            
        print("---------------")
        if choice == 1:
            content_model = ContentBasedFiltering.data_vectorizer()
            CollaborativeFiltering.read_csvs(True)
            collaborative_model = CollaborativeFiltering.algorithm_prepare()
    
            true_false = input(f"Save model? y/n: ")
            if true_false == "y" or true_false == "":
                    print("\033[32mSaving models...\033[m")
                    try:
                        ContentBasedFiltering.save_model(content_model)
                        CollaborativeFiltering.save_model(collaborative_model)
                        print("\033[32mModel saved!\033[m")
                    except Exception as e:
                        print(f"\033[31mError: {e}\033[m")


        elif choice == 2:
            movie_name = input("Insert your movie: ")
            test_input(movie_name)
            

main()