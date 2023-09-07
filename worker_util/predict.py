import load_model
import utility

class MSNovelist:
    def predict(mf, fp, filtered=True):
        try: 
            legality, composition_counter = utility.mol_form_processing(mf)
            if legality:
                results_df = utility.predict(composition_counter, mf, fp, filtered, load_model.k, load_model.model_encode, load_model.decoder)
                results_df['smiles'] = results_df['smiles'].str.replace('?', '')
                return results_df
            else: 
                print(f"The molecular formula {mf} does not fit into the list of allowed elements {utility.ELEMENTS_RDKIT}.")
        except Exception as e:
            print(e)