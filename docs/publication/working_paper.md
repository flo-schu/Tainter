# "We don't need no Administration..." or why trial and error increases survival of societies

## Abstract

In the past twenty years several events disrupted global economics and social wellbeing and generally shook the confidence in the stability of western societies. Popular examples are, the financial crisis, bankrupcy of multiple developed states, populism, war and climate refugees or Brexit. With this background we aimed to identify drivers of societal instability or even collapse. For this purpose a model was developed inspired by the theory of the collapse of complex societies. A simple network model simulated the development of complexity in terms of an administration body as a response to stresses affecting the productivity of the network agents.

We were able to illustrate societal collapse as a function of complexity measured in the share of administration in a network. Furthermore, we identified minimum requirements of the administration and the societal network topology to improve wellbeing of the society, estimated in terms of produced energy per capita. Finally we provide a mechanism for improving wellbeing and survival of the modeled society by enabling agents to randomly change between labor and administration, which is effective at very low rates.

## Model description


## Results

### Model of Tainter's theory of collaps of complex societies

The following figure shows the development of the societies administration along the time. Figure a illustrates Tainter's model of diminishing marginal returns of investments. Initially increases in administrator share result in relatively high increases in energy per capita production. After a short time the returns on investment are reduced, stagnate and when a tipping point is reached, any further increase in administrator share has adverse effects on the model society. The dashed lines show an analytic approximation of the model and confirm the underlying model dynamic. Figure b shows the same network modelled with a random exploration rate (i.e. change of node status A -> L or L -> A) of 1%. A low exploration rate visibly affects the survival and energy production of the society and converges to a stable fixpoint as shown by the analytic approximation.

![Tainter Model](../../results/model/20190327_1730/Admin_Ecap_twocases_analytic.png "Tainter Model")

Todo:
- plots:
  - ~~Administration share vs Ecap + analytische Lösung~~
  - Parameter grid (link density, efficiency, survived/energy)
  -
- parameter grids mit analytischer funktion reproduzieren
  - für survived / not survived
  - produced energy (integral ecap)
- Modellbeschreibung:
  - Gleichungen usw.
  - aussagekräftige Grafik
- lowest exploration rate herausfinden, die die überlebensdauer der netzwerke steigert
- berechnung t_crit: Zeitpunkt berechnen ab dem Ecap beginnt zu schrumpfen (Interpretation Tainter: Collaps)

<!-- ![survivaltime](../../plots/190222_1/survivaltime.pdf "grid plot parameters") -->

## Storyline

- Introduction
  - Warum ist die Tainter Theorie heute interessant

<!-- Möglichkeit A: Aufschreiben so wie es programmiert wurde.  -->
- Model Description
  - original Tainter Dynamics:
    1. Gesellschaft stirbt immer
    Aussagekräftige Grafik zur Modelldynamik
  - modified Tainter Dynamics:
    mit Exploration
  - Analytic approximation

- Results & Discussion
teil A
  - Exemplarische Ergenisse vom normalen Taintermodell (Plot)
  - Zusammenspiel Link Density, Efficiency "Conditions for an efficient administration"
    - Was sind realistische Effizienzraten und Vernetzungsgrade von Gesellschaften
    - Vergleich
    - Plot
  - makroskopische Näherung mit P_e = 0

teil B
  - das ganze noch mal mit Exploration
  - Main Message mit Plot
  - Erweiterung der makroskopischen Näherung

- Conclusion and Outlook


Arbeitsvorschlag:

Latex Dokument (Overleaf)
1. Abstract
2. Struktur
3. Stichpunkte für Kapitel
4. Abbildung + Bildunterschriften

Bis nächsten Freitag machen
