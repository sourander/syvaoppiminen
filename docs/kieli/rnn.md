---
priority: 710
---

# RNN ja jälkeläiset

## RNN

### Motivaatio

### Rakenne ja toiminta

### Unrolling

### Backpropagation Through Time (BPTT)

## Kehittyneemmät arkkitehtuurit

### LSTM (Long Short-Term Memory)

### GRU (Gated Recurrent Unit)

### Vertailu: RNN vs. LSTM vs. GRU

## Tehtävät

!!! question "Tehtävä: Sukunimien luokittelu Pt.1"

    Avaa Marimo Notebook `710_...py` ja tutustu koodiin. Kyseessä on [NLP From Scratch: Classifying Names with a Character-Level RNN](https://docs.pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html)-tutoriaali PyTorchin dokumentaatiosta, joka on käännetty Marimo-malliin sopivaksi. Suorita koodi ja tarkastele tuloksia.

    Mallin koulutus vei Macbook Pro:lla MPS:ää käyttäen 12 minuuttia.

!!! question "Tehtävä: Sukunimien luokittelu Pt.2"

    Palaa aiempaan `710`-alkuiseen Notebookiin. Kouluta malli uusiksi siten, että se ymmärtää myös suomalaisia sukunimiä. Käytä tässä apuna Avoidata.fi-palvelusta löytyvää Digi- ja väestäviraston julkaisemaa datasettiä [Väestötietojärjestelmän suomalaisten nimiaineistot](https://avoindata.suomi.fi/data/fi/dataset/none) (CC BY). Tee siis seuraavat:

    1. Lataa Excel-tiedosto
    2. Valitse kaikki sukunimet, joita on 700 tai yli
    3. Kopioi valitut leikepöydälle
    4. Liitä tiedostoon `data/names/Finnish.txt`

    Nyt sinulle pitäisi olla datasetissä uusi label `Finnish`, joka sisältää toista tuhatta suomenkielistä sukunimeä. Kouluta malli uudestaan, kenties eri tiedostonimellä, ja tarkastele tuloksia. Tunnistaako se sinut oikein? Mahdoitko olla training- vai test-datassa vai et kummassakaan?

## Lähteet