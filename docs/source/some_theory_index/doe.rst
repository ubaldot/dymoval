Design of Experiments (DoE)
===========================

When running experiments, it is important to stimulate the target environment
in a way that we can extract
as much information as possible from it.

Good experiments shall stress the target environment as much as possible under
different conditions.
This is important because a model is as trustworthy as the measurements
dataset used for validating it is *informative*.

    **Example**

    If we are developing the model of a car, we want to log sensors
    measurements while driving within a wide range of speeds, with different
    accelerations profiles, in different road and weather conditions and so on
    and so forth.

    If we log sensors measurements only when we are driving on a flat road and
    in the range 0-10 km/h and by doing exactly the same maneuvers over and
    over, then it would be hard to disagree on that the collected measurements
    dataset is poorly informative.



In the current release, *Dymoval* only stores the coverage region of the
measurements dataset and compute some statistics on it.
In this way, the user have an understanding under which conditions the
developed model is trustworthy, provided that the validation results provides
good figures.

.. note::
  In future releases we plan to further provide measures (Cramer-Rao Theorem?
  Fisher Matrix?) on the
  information level contained in a given measurements dataset in within its
  coverage region.


It is worth noting that a measurements dataset covering a fairly large region
won't necessarily imply *information richness.*
This happens for example when you take a wide range of values but you
stimulate your target environment only with constant inputs in within such a
range.  You would certainly have a measurements dataset with a fairly large
covered region but it would contain little information.

    **Example**

    With reference to the example above, imagine driving the car only at
    constant speeds ranging from 0 to 180 km/h on a flat road, without ever
    accelerating or braking. For instance, you make a first run driving (and
    logging) data at a constant speed of 10 km/h, without accelerating or
    braking, and staying on a flat road. Then, you perform a second run at a
    constant speed of 20 km/h under the same conditions as the previous run,
    and so on, until reaching the final run at 180 km/h.

    Your measurements dataset will have a fairly large coverage region, but it
    will contain little information since all the runs were conducted at
    constant speeds without any acceleration or braking, and on a flat road.


How can you design experiments that produce sufficiently informative
measurement datasets?

Well, this issue cannot be fully automated, but there is some theory behind it
in the field of Design of Experiments (DoE). Feel free to search for more
details on this topic.

Due to the fact that DoE cannot be automated, it will not be included in
Dymoval, at least for now.

.. vim: set ts=2 tw=78:
