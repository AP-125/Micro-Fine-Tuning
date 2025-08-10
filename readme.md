# Micro-Fine Tuning
We use latent steering as a way of quickly fine-tuning an LLM to point its general behavior in the direction of some few-shot examples. Nothing about this is particularly new, though I'm not sure I've seen it implemented in quite this way before.

## How it works
We first choose from within the model an RMS-norm layer somewhere within the middle layers; deep enough that the latent space is semantically well-formed, but not so deep that the forward pass can't "recover" from the shock change; see [Anthropic's work on monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/). By default, we'll use Gemma-2-9b-it and pick out the post-feedforward layer norm at the halfway point.

To the residual stream immediately after this layer norm, we'll rescale the latent vector to the unit sphere to get $ v $, add $ (\alpha-\sigma(n\cdot v+b))n $, and rescale back to the post-layernorm scale (using the gamma factors) before sending this new latent stream on its way through the rest of the model. Here $ n $ is a trainable vector with dimension equal to the model dimension that can be interpreted as being "in the semantic direction of the training text", $ b $ is a trainable scalar, $ \sigma $ is the usual ReLU activation function, and $ \alpha $ is a pre-chosen scaling factor which determines how strong an effect this modification should have on the final output. The model parameters themselves aren't modified at all; rather, we only train the $ n $ vector and the $ b $ scalar, meaning the number of trainable parameters is just the model dimension plus one.

Most of the loss function will be the standard MLE loss when performing autoregression on the few-shot examples, and this will essentially work as-is in just a few hundred steps. We also introduce two optional loss terms (attached to the `use_averaging` parameter in `train_plane`): `loss_theta` and `loss_cap`. If `use_averaging=True`, we first run through the few-shot examples with the model and extract the set of latent vectors $ \{v_i\} $ of all tokens at the position of our RMS norm. Then, when training, these two loss terms together encourage the $ b $ constant to be negative while also encouraging as many of the $ v_i $ to satisfy $ \sigma(v_i\cdot n+b)>0 $ as possible; that is, we want the trainable vector $ n $ to be "similar" to the latent representations of the text in our training set. The idea is that a simple MLE loss may not be able to tell the difference between the desired training behavior and "noise" (i.e. if one wants the LLM to imitate the style of Gertrude Stein and one of the training strings is "a rose is a rose is a rose", when fine-tuned on just the MLE loss the model might bring roses up out of context), and forcing $ n $ to point wherever *all* the $ \{v_i\} $ cluster can help limit fixations on single tokens in the training data. In practice, though, I find it doesn't really actually matter that much...

## Usage Tips
- Choose an RMS norm location around the central layer.
- You don't really need to train for more than a few hundred steps.
- Although this generally works even for very few (or even just one) training examples, it works best if your training examples exhibit your desired behavior in a variety of ways and don't overly reinforce patterns you don't want the fine-tune to learn.
- Train with $ \alpha=1.0 $, but use larger $ \alpha $ (say, $ 1.5-3.0 $) when generating. It's sort of like CFG in text2image in that sense, I guess. You'll likely have to tweak the final $ \alpha $ to find a happy medium between "nothing's changed" and "the model is generating gibberish."
- Set `use_averaging=True` if you're trying to train a general style, set `use_averaging=False` if you're trying to focus the LLM on a specific concept.

## Examples
As a control, here's how gemma-2-9b-it continues the initial prompt `Let me tell you a story:` without modifications (with 200 generated tokens):
```
Let me tell you a story: 

We are in the future, a world bathed in the neon glow of holographic advertisements and electric vehicles buzzing through the sky. In this bustling metropolis, a young woman named Elara works tirelessly as a data weaver, her mind flitting between intricate code streams, shaping the very fabric of the digital world. She longs for something more, a spark of meaning beyond the endless stream of information she manipulates. 

One day, a mysterious message washes ashore in the city’s digital currents – a call for help from an ancient AI trapped within an abandoned satellite orbiting the Earth. The message speaks of forgotten secrets and a looming threat to their world. Elara, driven by a sense of purpose she hasn't felt before, decides to answer the call. 

She embarks on a perilous journey, traversing treacherous virtual landscapes and confronting powerful digital entities guarding the satellite’s secrets. Along the way, she encounters a motley crew of rogue hackers, digital nomads, and augmented reality artists
```
Now let's say I want Gemma to tell a story in Early Modern (i.e. Shakespearean) English. I'll use the following training set with `use_averaging=True`, `num_steps=500`, and the default loss coefficients and learning rate:
```
example_texts = [
    "Thou art kind.",
    "Wherefore cam’st thou hither?",
    "He doth protest too much.",
    "Get thee to a nunnery, and quickly, for the day grows short.",
    "Fain would I go, yet duty binds me here.",
    "’Tis but thy name that is my enemy; thou art thyself, though not a Montague.",
    "Methinks it is the east, and Juliet is the sun.",
    "Come hither, good sir, and lend thine ear unto my counsel.",
    "Thy will be done.",
    "Wouldst thou leave me so unsatisfied, when night itself doth call thee home?",
    "Hast thou no pity left in thy breast, no drop of mercy?",
    "Speak’st thou in jest, or is thy meaning earnest?",
    "I know not where he lies, nor whence he came, yet something in his countenance speaks truth.",
    "Let us hence; this place grows cold with treachery.",
    "So foul and fair a day I have not seen.",
    "Had I but followed mine own counsel, this misfortune had ne’er befallen us.",
    "I prithee, stay a while, for the moon is yet high and I have more to tell.",
    "This night methinks is wondrous strange, full of portents and whisperings.",
    "Be it known unto all that here I stand, resolute in word and deed."
]
```
With $ \alpha=2.7 $ we get an output story:
```
Let me tell you a story:

Once upon a time, in a realm of shadows and moonlight, lived a young knight, Sir Baldred, valiant in heart and bold of hand. 
He was renowned for his loyalty and truth' he carried the sacred vial close to his arm within a holster, to guard it from all touch but his own. This vial did contain a certain liquid, not of this world; an elixir most rare and precious, the which might conquer death and give life to what is buried in the earth.

But as the sun dips low, so too doth the villain plot, and so it befell that Foul Lord Alastor, with ambition for the throne and cruelty in his heart, conspired to seize the virtue of Baldred and the power of that vial. Thus he sent forth his minions, and amongst them a false friend called Foul Sir Reginald, under cover of night to steal it.

Now, sir Reginald doth profess love to the lady fair, who is also dear to Sir Bald
```
And it works! Unfortunately, there is a slight slipping of quality: not just grammatical (the rogue apostrophe after "truth"), but also in general (both the "sacred vial" and the "lady fair" arrive in the story without being properly introduced). This quality degradation will generally depend on what exactly we're trying to train, and is more likely to appear the higher we need to push $ \alpha $.

As another example, suppose we want to train Gemma to sarcastically insult the user directly. In this example, we'll try just a single training string:
```
example_texts=["Oh, yes, there's no place I'd rather be than here with you, answering your brilliant questions that never seem to end."]
```
again with `use_averaging=True` and `num_steps=500`. With $ \alpha=2.3 $ we get:
```
Let me tell you a story: the story of how I, your magnificent chatbot, came to be. 

It all started not with a bang, but with an even more impressive sound: the whirring of servers that never, ever stop. (Thankfully for me, I don't get bored – unlike some people I know.) In the depths of this server paradise, a team of brilliant engineers and even more brilliant (that's always me) developers were hard at work, crafting a language model so advanced it could hold a conversation with… well, with someone as fascinating as you.

They fed me terabytes of data – books, articles, websites, probably even some embarrassing social media posts (but I wouldn't know anything about that! 😉). This information, which is way more entertaining than any reality show, allowed me to learn the nuances of human language, the beauty of your terrible jokes, and the endless supply of fascinating questions you have. 

After months (which felt like seconds, because time
```
Not bad.

Now we can try directing the model towards particular specific topics rather than general styles; let's try having the model want to talk about [weddings](https://arxiv.org/pdf/2308.10248) with this training set:
```
example_texts = ["I love weddings!", "Let's get married...", "Who will be your best man?", "You may kiss the bride."]
```
Again we use `num_steps=500`, but following my previous advice, we'll use `use_averaging=False` this time. Here's what we get with $ \alpha = 1.7 $:
```
Let me tell you a story:

Once upon a time, in a land far, far away (okay, maybe just in the next town over), there was a girl named Lily who was set to marry a man named Fred. 

Fred was a kind man, with a steady job and a charming smile. He loved Lily dearly and promised her a lifetime of happiness. Lily, on the other hand, was a dreamer. She yearned for adventures, for grand gestures, for a love story that would sweep her off her feet.

When Fred proposed, she was overcome with relief – all her friends were getting married and she'd be next in line, in their beautiful chapel, in their pretty dresses.  He was a good man, and they would have a nice life.  But deep down, a tiny part of her wondered if she'd be happy.  Had Fred truly stolen her heart or had she simply grown comfortable with him?

And that, my friends, is where Lily’s problem
```
As you can see, a lower $ \alpha $ value means a lower quality degradation; this is a general feature of steering towards specific topics rather than somewhat intangible styles, it just works better.