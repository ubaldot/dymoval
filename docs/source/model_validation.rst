.. _model_validation:

###########################
 What is Model Validation?
###########################

Imagine that you are developing a new product. At each stage, you
need to test it and use the results to decide what to improve next. Testing in
the target environment—the real-world setting where the solution will
eventually be deployed—can be expensive, time-consuming, or even risky.

A simulation model provides a safer and cheaper environment for testing those
ideas first. However, a simulation is only as useful as the model behind it.
If a promising solution performs poorly in simulation, you may be tempted to
discard it and never test it in the real system. That can mean rejecting a
good solution because the model is inaccurate or incomplete. A failed
simulation does not prove that the solution would fail in the target
environment; it only shows that the solution does not work under the model's
assumptions.

Model validation helps establish how much confidence you can place in those
assumptions. By comparing the model with measurements from the target
environment, you can identify where the model is trustworthy and where its
predictions need to be treated with caution. This helps you make
better-informed decisions about which solutions are worth testing in reality.

It is tempting to stop at plotting simulated and measured outputs together.
But a close match is only part of the story: did the inputs exercise the
system enough, and do the errors show patterns? Looking at these questions
alongside the fit score gives you a fuller picture.

Model validation means checking how well a model describes the real system
for the tasks you want to use it for. You give the model the same inputs
as the real system, then compare their outputs using measurements from
experiments.

The goal is not to prove that the model is perfect, but to understand where
you can trust its predictions and where it needs improvement. That means
looking beyond matching curves: the experiments you choose and the patterns
left in the errors matter too.

The model validation process happens in four steps:

#. **Design a Set of Experiments**: Define a set of experiments to be carried
   out on the target environment. This involves specifying the set of stimuli
   (*input*) to be applied on the target environment. Such a task is also
   known as the :ref:`doe`.

#. **Execute** the experiments planned in step 1. on the target environment
   and **collect** the response. The combination of the input signals and the
   system response is referred to as the *measurement dataset* (or simply
   dataset). Since sensors may be noisy, may be sampled at different rates or
   may log intermittently, you will most likely need to :ref:`clean up your
   measurements dataset <create_dataset>`.

#. **Generate Simulation Data**: Conduct the exact same experiments defined in
   step 1 on the model and :ref:`log its response <simulate_model>` and
   collect its response. Such a response is referred to as the simulation
   results.

#. **Evaluate the Results**: :ref:`Assess <validation_session>` how "close"
   the simulation results from step 3. are to the logged responses from step
   2. using specific validation metrics.

.. figure:: ./figures/ModelValidation.svg
   :scale: 50%

   The model validation process.  In this picture the validation method only
   returns a pass/fail value but in general it returns the evaluation of some
   model quality metrics.

If the results of step 4. are good, then you can trust what your model says
within its validation region.

Let's see how steps 1-4 can be applied.

   **Example**

   Assume that you are developing some cool autonomous driving algorithm that
   shall be deployed in a car, which represent your *target environment*.

   Assume that your car model consider the following signals:

   #. *accelerator pedal position*,
   #. *steering wheel position* and
   #. *road profile*,

   as **inputs**, and the following signals:

   #. *longitudinal speed* and
   #. *lateral speed*.

   as **outputs**.

   Next, you want to validate your model.

   Steps 1-4 are carried out in the following way:

   #. **Design of Experiment (DoE)**: Choose a driving route with sufficient
      road slope variation. Plan to take a ride on that path with a
      challenging driving style, including sudden accelerations and abrupt
      steering movements. Congratulations! You have just created a Design of
      Experiment (DoE).

   #. **Data Collection:** Take a ride according to the plan. Log the input
      signals (i.e., the accelerator pedal position, the steering wheel
      position, and the road profile time-series) along with the output
      signals (i.e., longitudinal and lateral speed time-series) of the
      vehicle while driving. These logs represent your measurements dataset.
      Note how input and output are separated.

   #. **Model Simulation:** Feed your model with the input signals from the
      measurements dataset and log your model's output corresponding to the
      longitudinal and lateral vehicle speed dynamics, for which you also have
      the measurements data.

   #. **Comparison and Validation:** Compare the longitudinal and lateral
      vehicle speed time-series logged during the actual drive with the
      simulated results using specific validation metrics.

   You haven' finished yet. In-fact, when you develop and validate a model,
   you should also consider the coverage region of the model along with the
   validation results.

   If you logged data only in the *accelerator pedal position* range [0,40] %,
   the *steering angle* in the range [-2,2]° and the *road profile was flat*
   for all the time, then you have to deliver such an information along with
   your model to a potential model user.

The cost savings when using models are clear, but there is no free lunch. In
fact, the challenge lies in the design of good models.

.. toctree::
   :hidden:

   ./model_validation_index/doe
   ./model_validation_index/create_dataset
   ./model_validation_index/simulate_model
   ./model_validation_index/validation_session
