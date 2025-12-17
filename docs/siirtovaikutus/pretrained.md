---
priority: 600
---

# Koulutetun mallin käyttö

Tässä osiossa käydään läpi, miten esikoulutettua mallia voidaan hyödyntää omassa projektissa ilman, että tarvitsee kouluttaa mallia alusta asti itse. Tämä on esivaihe seuraavalle osalle, jossa käsitellään siirtovaikutusta (transfer learning). Lisäksi kokeilemme, voiko esikoulutettua mallia käyttää sellaisenaan luomaan piirrevektoreita, joiden pohjalta voidaan kouluttaa Johdatus koneoppimiseen -kurssilta tuttuja perinteisiä koneoppimismalleja, kuten Random Forestia tai Logistic Regressionia.

## PyTorchin mallit

PyTorchin valmiiksi koulutettuihin malleihin pääsee käsiksi kahta reittiä:

* [PyTorch Hub](https://pytorch.org/hub/) -sivuston kautta
* `torchvision.models` -moduulin avulla

Jos tarkkoja ollaan, niin nämä reitit ovat sinänsä samat, että `torchvision.models` käyttää taustalla PyTorch Hubia. Erona on, että `torchvision.models` tarjoaa vain kuvantunnistukseen tarkoitettuja malleja, kun taas PyTorch Hubista löytyy malleja monenlaisiin tarkoituksiin, kuten luonnollisen kielen käsittelyyn (NLP) ja generatiivisiin malleihin. Jotta tämä ei olisi liian helppoa, niin monet näistä malleista ovat fyysisesti säilöttynä Hugging Facen mallivarastoon. Katso vaikka [PyTorch Hub: PyTorch-Transformers](https://docs.pytorch.org/vision/main/models)-dokumentaatio.

TODO

TODO

## Hugging Face

PyTorchin itsensä tarjoamien esikoulutettujen mallien lisäksi on olemassa useita kolmannen osapuolen sivustoja tai kirjastoja, jotka tarjoavat laajan valikoiman esikoulutettuja malleja eri tarkoituksiin. Eri julkaisuihin löytyviä malleja voi löytyä esimerkiksi Githubista tai Zenodosta. Tällä kurssilla käsitellään yksinkertaisuuden vuoksi vain ja ainoastaan Hugging Facea. Mikä siis on Hugging Face? Ei ainakaan sitä, mikä se perustamishetkellä oli:

> "In 2016, Clement Delangue, Julien Chamound, and Thomas Wolf launched an AI-powered chatbot for teenagers called Hugging Face. Their vision for the product was a digital friend that was entertaining enough for people to have fun talking to it"
>
> — [Jason Shen, 2024][^pathtopivot]

Nykypäivänä Hugging Face on API-talouden ilmentymä. Sen etusivun navigaatiossa olevista osioista voi päätellä sen eri palvelut:

* **Models**: Esikoulutettuja malleja (2.2M kirjoitushetkellä, joista 220k PyTorch-malleja). 
    * :arrow_up: Tämän tunnin aihe!
* **Datasets**: Valmiita datakokoelmia.
* **Spaces**: Mahdollistaa esimerkiksi Gradio‑ tai Streamlit‑pohjaisten sovellusten julkaisemisen ja jakamisen.
* **Community**: Blog articles, Social posts, Daily papers.
* **Docs**: dokumentaatio.
* **Enterprise**: yrityspalvelut.

Sinun tulee rekisteröityä Hugging Facen käyttäjäksi, jotta voit ladata malleja ohjelmallisesti. Rekisteröityminen on ilmaista ja peruskäyttö on ilmaista. Joitakin malleja saa käyttää *vain jos* pyydät käyttöluvan sen tekijältä. Näistä käytetään termiä [Gated models](https://huggingface.co/docs/hub/en/models-gated).

Erityisen mielenkiintoinen on [PyTorch Image Models (timm)](https://huggingface.co/timm)-organisaatio Hugging Facessa, ja siihen liittyvä [timm-kirjasto](https://pypi.org/project/timm/). Tämä kirjasto tarjoaa yli 600 esikoulutettua mallia kuvantunnistukseen, mukaan lukien monet viimeisimmät arkkitehtuurit.

## Tehtävät

!!! question "Tehtävä: Luo Hugging Face tunnus"

    Rekisteröidy Hugging Facen käyttäjäksi osoitteessa [Hugging Face](https://huggingface.co/. Suosittelen käyttämään henkilökohtaista tunnusta koulun sähköpostin sijasta. On todennäköistä, että haluat käyttää palvelua myös valmistumisen jälkeen ja yksityisissä projekteissa.

    **Kun olet rekisteröitynyt**, sinun tulee luoda [Access Token](https://huggingface.co/settings/tokens) -avain. Avainta käytetään autentikointiin ohjelmallisesti (eli Pythonissa.) Tokenin luontiin löytyy ohjeistusta [User access tokens](https://huggingface.co/docs/hub/en/security-tokens)-dokumentaatiosta. Lyhyt ohje on kuitenkin:

    * Klikkaa profiilikuvaketta oikeasta yläkulmasta ja valitse "Access tokens".
    * Klikkaa `+Create new token` -painiketta.
    * Valitse nimi ja haluamasi *fine-grained* -oikeudet (esim. `read` riittää tähän kurssiin). Älä stressaa oikeuksista: voit aina luoda uuden avaimen jos tarvitset enemmän oikeuksia.

    **Kun olet luonut tokenin**, ota se väliaikaisesti talteen. Tässä välissä voit tehdä autentikaation `uvx`:n ja `hf`-kirjastojen avulla:

    ```python
    uvx hf auth login
    ```

    Komento kysyy sinulta tokenia. Liitä token tähän. Muut kysymykset ja niiden tarkemman kuvauksen löydät [CLI: hf auth login](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli#hf-auth-login)-dokumentaatiosta.

!!! question "Tehtävä: Hugging Face Hello World"

    Asenna transformers-kirjasto, jos sinulla ei sitä jo ole. Suorita seuraava komento projektisi juuressa:

    ```bash
    uv add "transformers[retrieval,timm,torch-vision]"
    ```

    !!! warning

        Tässä pitää olla tarkkana kirjastojen riippuvuuksien suhteen. Tarvitsemme kirjastot `datasets` ja `huggingface_hub` myös, mutta niitä ei kannata lähtökohtaisesti asentaa erikseen. Asenna ne sen sijaan käyttäen `transformers`-kirjaston **extras**-ominaisuutta eli hakasulkeissa listatut lisäosat. Tämä varmistaa, että asennettavat versiot ovat yhteensopivia keskenään.
        
        Kirjoitushetkellä minulle ei esimerkiksi ollut ollenkaan `torch`-kirjastoa asennettuna omana rivinään, koska `transformers` hoiti sen asennuksen automaattisesti (koska extrat). Nämä asennettujen pakettien suhteet selviää komennolla `uv tree`, ja vielä tarkemmin voi tutkia `uv tree --package=transformers`. Rankasti parsittu output, joka korostaa mahdollista ongelmaa, näkyy alla:

        ```
        transformers v4.57.3
        ├── huggingface-hub v0.36.0
        │   └──  ...
        ├── tokenizers v0.22.1
        │   └── huggingface-hub v0.36.0 (*)
        ├── datasets v4.4.1 (extra: retrieval)
        │   └── huggingface-hub v0.36.0 (*)
        ├── timm v1.0.19 (extra: timm)
        │   ├── huggingface-hub v0.36.0 (*)
        │   ├── torch v2.9.1
        │   │   └── ...
        │   └── torchvision v0.24.1
        │       └── torch v2.9.1 (*)
        └── torchvision v0.24.1 (extra: torch-vision)
        ``` 

        Jos käyt [gh:huggingface/huggingface_hub](https://github.com/huggingface/huggingface_hub)-repossa, huomaat, että `huggingface_hub`-kirjasto on tuoreelta versioltaan (kirjoitushetkellä) `v1.2.3`. Kyseinen kirjasto on loikannut uuteen Major-versioon Oct 27. Jos asentaisit sen käsin, `uv` asentaisi tämän tuoreen version, ja **major on taaksepäin yhteensopimaon by definition**. Saisit paljon erroria. Summa summarum, käytä ison kirjaston extra-asennuksia aina kun mahdollista!

        1. Älkää epäröikö kysyä apua, jos asennuksessa ilmenee ongelmia!
        2. Älä vaivu *"opettajan pitäisi hoitaa nämä asiat"* -ajatteluun. Tämä on osa ohjelmistokehityksen arkea, ja on tärkeää oppia ratkaisemaan nämä ongelmat. Et todellakaan tule säästymään kirjastoriippuvuuksien kanssa painimiselta myöhemminkään urallasi.

    Avaa `600_hello_hugging_face.py` ja suorita se. Jos autentikointi on onnistunut, ohjelma lataa Hugging Facen `transformers`-kirjatoa käyttäen sentimenttianalyysiin soveltuvan mallin ja suorittaa ennusteen esimerkkilauseelle. Kirjoitushetkellä malli on [distilbert/distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english), joka on esikoulutettu DistilBERT-malli, hienosäädetty SST-2-datalle (Stanford Sentiment Treebank). Malli on noin 270 MB kokoinen.
    
    Ohjelman pitäisi tulostaa jotain seuraavan kaltaista:

    ```
    [{'label': 'POSITIVE', 'score': 0.9998}]
    ```

!!! question "Tehtävä: Hate Speech over Naive Bayes"

    Toteuta `601_hate_speech.py`-tiedostoon puuttuvat kohdat. Tutustu skriptin toimintaan. Jos sinulla on Johdatus koneoppimiseen -kurssin muistiinpanot saatavilla, tarkista, mihin tarkkuuteen Naive Bayes -malli ylsi saman datan ja ongelman kanssa.

## Lähteet

[^pathtopivot]: Shen, J. *How Hugging Face Transformed a $4.5B AI Powerhouse [Pivot Case Study]*. The Path to Pivot. 2024. https://www.pathtopivot.com/hugging-face-pivot-case-study/