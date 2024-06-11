from .preprocess import pre_processing
from .bert import bert
from .sentence import SimilarityFunctions, SentenceEmbeddings

def evaluate(question,key,answer):
    # Preprocess student answer
    pp_desired = pre_processing(question, key)
    pp_student = pre_processing(question, answer)
    
    if len(pp_desired) == 0 :
        pp_student = pre_processing('', answer)
        pp_desired = pre_processing('', key)
    elif len(pp_student) == 0:
        return 0
    
    print('preprocessed student answer')
    word_array_1 = bert(pp_desired)
    word_array_2 = bert(pp_student)

    print('bert embeddings done')
    # Compare and assign cosine similarity to the answers

    similarity_tools = SimilarityFunctions(word_array_1, word_array_2)
    sentence_embed = SentenceEmbeddings()

    text_1_embed = sum(word_array_1)
    text_2_embed = sum(word_array_2)

    print('embedding added')
    bert_similarity_score = similarity_tools.get_cosine_similarity(text_1_embed, text_2_embed)
    return bert_similarity_score