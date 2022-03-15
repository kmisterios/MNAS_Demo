from flask import (
    Flask,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    send_file,
    send_from_directory,
    session,
    url_for,
)
from config import Config
import pickle
from web_scraping import manual_news_scraping
from utils.utils import load_and_prepare_data, prepare_for_training, preprocess, compose_pairs, read_news_to_csv, df_to_dict
from model.train import train_eval
from threading import Thread
import flag



app = Flask(__name__)
app.config.from_object(Config)

LANG_DICT = {"ru" : flag.flag("RU"), "en" : flag.flag("US"), "es" : flag.flag("ES"),
             "de" : flag.flag("DE"), "fr" : flag.flag("FR")}


def compare_news():
    path = "./news.pkl"
    data = read_news_to_csv(path)
    dataset = compose_pairs(data)
    data_eval = load_and_prepare_data(dataset, preprocess)
    data_eval = prepare_for_training(data_eval, inference = True)
    config_path = "./model/models_configs_HF_FC_L2NORM.pickle"
    with open(config_path, "rb") as f:
        model_configs = pickle.load(f)
    result_FC_L2Norm_cosim1 = {}
    checkpoints_path = "./model/checkpoints"
    figs_path = "./figs"
    model_name = 'xlm-mlm-17-1280'
    data_result = train_eval(model_name, data_eval, model_configs[model_name]["batch_size"], model_configs[model_name]["batch_size_val"], 
                model_configs[model_name]["linear_layer_size"], model_configs[model_name]["num_epoch"],
                result_FC_L2Norm_cosim1, train = False, checkpoints_path = checkpoints_path, figs_path = figs_path, inference = True)

    path = "./res"
    df_to_dict(data_result, path)

    


@app.route("/")
def route():
    return redirect("/search")


@app.route("/search", methods=["GET", "POST"])
def search():
    if request.method == "POST":
        search_text = request.form.get("search_text")
        session["search_text"] = [search_text]
        # scrap the news
        manual_news_scraping("./news", search_text, 1)
        # analyse them
        compare_news()
        return redirect(url_for(".search_request", search_text=search_text))
    return render_template("search.html", title="Search")


@app.route("/search/results", methods=["GET"])
def search_request(): 
    res_path = "./res.pkl"
    with open(res_path, "rb") as f:
        res = pickle.load(f)
    search_text = session.get("search_text", None)[0]
    res["query"] = search_text
    return render_template("results.html", res=res, lang_dict = LANG_DICT)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)
    # app.run()
