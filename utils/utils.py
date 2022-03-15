import pandas as pd
import numpy as np
import pickle


def df_to_dict(data, path):
    n = data.shape[0]
    res = {"news" : [], "news_count" : n}
    for i in range(n):
        line = data.loc[i]
        element = {}
        element["content1"] = line["title1"] + "\n\n" + line["text1"]
        element["content2"] = line["title2"] + "\n\n" + line["text2"]
        element["distance"] = round(line["Overall"], 3)
        element["lang1"] = line["lang1"]
        element["lang2"] = line["lang2"]
        res["news"].append(element)
    with open(path + '.pkl', 'wb') as file:
        pickle.dump(res, file, pickle.HIGHEST_PROTOCOL)


def read_news_to_csv(news_path):
    with open(news_path, "rb") as f:
        res = pickle.load(f)
    res = res[0]
    data = pd.DataFrame(columns = ["lang", "title", "content", "url"])
    for key in res:
        for news in res[key]:
            d = {"lang": key} 
            d["title"] = news["headline"]
            d["content"] = news["content"]
            d["url"] = news["url"]
            data = pd.concat([data, pd.DataFrame.from_records([d])])
    data.reset_index(inplace = True, drop = True)
    return data


def compose_pairs(df):
    columns = ["lang1","title1", "text1", "url1", "lang2", "title2", "text2", "url2"]
    data = pd.DataFrame(columns = columns)
    traget_news = df[df["lang"] == "en"].iloc[0]
    langs = set(df["lang"])
    for lang in langs:
        news_to_compare = df[df["lang"] == lang]
        index = np.random.randint(len(news_to_compare))
        p_news_to_compare = news_to_compare.iloc[index]
        df_tmp = pd.concat([traget_news, p_news_to_compare], axis = 0)
        df_tmp.set_axis(columns, axis=0, inplace=True)
        data = pd.concat([data.transpose(), df_tmp], axis = 1, ignore_index=True).transpose()
    return data


def preprocess(document, method = "HF_emb_FCL2NORM"):
    if method == "HF_emb_FCL2NORM":
        document = document.replace("\n", " ")
        words = document.split(" ") # Tokenize
        words = words[:512]
        return " ".join(words)
    
    return document



def load_and_prepare_data(dataframe, preprocess_func = None, content = ['title', 'text']):
    df_all = dataframe.copy()
    df_all.reset_index(drop = True, inplace = True)
    df_all.fillna(value = "", inplace=True)
    if len(content) > 1 :
        df_all["content1"] = df_all[['title1', 'text1']].apply(lambda x: x[0] + ' ' + x[1], axis=1)
        df_all["content2"] = df_all[['title2', 'text2']].apply(lambda x: x[0] + ' ' + x[1], axis=1)
    else:
        df_all["content1"] = df_all['title1'].values
        df_all["content2"] = df_all['title2'].values
    if preprocess_func is not None:
        df_all["content1"] = df_all["content1"].apply(preprocess_func)
        df_all["content2"] = df_all["content2"].apply(preprocess_func)
    return df_all


def prepare_for_training(df, method = "HF_emb_FCL2NORM", inference = False):
    if method == "HF_emb_FCL2NORM":
        if not inference:
            df['label'] = df['Overall'].apply(lambda x: (x - 1)/3)
        df["content"] = df[["content1", "content2"]].apply(lambda x: [x[0],x[1]], axis=1)
    if method == "HF_emb_FCReg":
        if not inference:
            df['label'] = df['Overall'].apply(lambda x: (x - 1)/3)
        df['content1'] = df['content1'].apply(lambda x: x[:500])
        df['content2'] = df['content2'].apply(lambda x: x[:500])
        df['content'] = df[['content1', 'content2']].apply(lambda x: ' '.join([x[0], '[SEP]', x[1]]), axis=1)
        df.reset_index(drop = True, inplace = True)
    if method == "NLI":
        df['content1'] = df['content1'].apply(lambda x: x[:200])
        df['content2'] = df['content2'].apply(lambda x: x[:200])
    return df