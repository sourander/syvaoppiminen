---
priority: 210
---

# PyTorch 101

Tässä otetaan PyTorch tutuksi. Perusjuttuja. Muutama harjoitus, kuten Sigmoid ja ReLu implementointi käyttämättä PyTorchin valmiita funktioita.

## TODO

Tähän teoriaa hieman.


## Tehtävät

!!! tip "Muokkaa ja kokeile vapaasti!"

    Ennen tehtävien alustamista haluan välissä haluan huomauttaa, että **on täysin sallittua** muokata olemassaolevia Notebookeja ja/tai luoda omia Marimo-kirjoja, joissa kokeilet PyTorchin toiminnallisuuksia.

    **Ole rohkea!** Kokeile, tutki ja muokkaa. Riko ja korjaa.
    
    Dokumentoi löydöksesi oppimispäiväkirjaan.

!!! question "Tehtävä: Tutustu NumpyNNwithBCE -malliin"

    Avaa `210_numpy_to_pytorch.py`-tiedosto ja tutustu `PyTorchNN`-malliin. Malli on sama 2-2-1 kun aiempi `NumpyNNwithBCE`-malli, mutta toteutettu PyTorchilla. 
    
    Aja koodi ja tutki mitä tapahtuu. Varmista, että ymmärrät, kuinka mikäkin rivi koodia liittyy tähän mennessä kurssilla opittuun.

    Keskity erityisesti PyTorch-kirjaston tensori- ja mallitoiminnallisuuksiin, joita Marimo-notebookissa käytetään.

!!! question "Tehtävä: PyTorch Learn the Basics: Tensors"

    Avaa `211_pytorch_tensors.py`. Huomaa, että kyseessä on PyTorchin virallinen [Learn The Basics: Tensors](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html) -opas, joka on käännetty Marimo-muotoon. 
    
    Jos käytät Google Colabia, voit avata alkuperäisen ohjeen.

!!! question "Tehtävä: PyTorch Introduction to Pytorch Tensors"

    Avaa `212_tensors.py`. Huomaa, että kyseessä on PyTorchin virallinen [Introduction to PyTorch - YouTube Series: Introduction to PyTorch Tensors](https://docs.pytorch.org/tutorials/beginner/introyt/tensors_deeper_tutorial.html) -opas, joka on käännetty Marimo-muotoon.

    Myös tässä voit käytätä Google Colabia alkuperäisen ohjeen avaamiseen tai noudattaa kurssin Marimo-versiota.


!!! question "Tehtävä: Auto MPG"

    Avaa `213_auto_mpg.py`. Notebookissa on matalan kynnyksen käyttöönotto PyTorch-mallille. Data on loppumetreille asti aiemmin tutussa Pandas DataFramessa. Seuraavilla viikoilla tutustumme paremmin esimerkiksi Dataset ja DataLoader -toiminnallisuuksiin. Keskitytään toistaiseksi mallin kouluttamiseen ja tulosten validointiin yksinkertaisella MAE-metriikalla.
    
!!! question "Tehtävä: Kyberviha PyTorch-mallilla"

    Johdatus Koneoppimiseen -kurssin logistisen regression tehtävänä oli tunnistaa, onko henkilö kokenut kybervihaa viimeisen vuoden aikana.

    Alkuperäinen datasetti löytyy Data in Brief [Digital skills among youth: A dataset from a three-wave longitudinal survey in six European countries](https://www.sciencedirect.com/science/article/pii/S2352340924003652)-data-artikkelista. Käytämme kuitenkin vertailun vuoksi `ySKILLS_longitudinal_dataset_teacher_processed.csv`-tiedostoa, joka on esiprosessoitu versio alkuperäisestä datasta. Esikäsitellyn datan voi ladata [gh:sourander/ml-perusteet-code](https://github.com/sourander/ml-perusteet-code)-repositoriosta `data/yskills`-hakemistosta.

    Muistanet, että tulos oli kohtalaisen heikko. Tämän harjoituksen motiviina on tutkia, ovatko neuroverkot hopealuoti, joka parantaa tuloksia merkittävästi – vai käykö kenties niin, että joudut taistella hyperparametrien kanssa saadaksesi edes jossain määrin vertailukelpoisen tuloksen.

    !!! warning

        Tämän tehtävän linkillä on vahva deprekoitumisvaroitus. Johdatus Koneoppimiseen -kurssi refaktoroidaan 2026 keväällä. Jos linkit eivät toimi, ota yhteyttä opettajaan. Hän on unohtanut päivittää linkit.
