# Contributing to speakmin project
This file provides general guidance for anyone contributing to speakmin project. Adding new features, improving documentation, fixing bugs, writing new tests, designing and coding new examples or writing tutorials are all examples of helpful contributions.

## Prerequisits
If you are new to GitHub, you can find useful documentation [here][1], and information on the `git` version control system in this [handbook][2].

## Steps to contribute
When contributing to speakmin project, we ask you to follow some simple steps:

1. Create a new git issue
	- describing the type of contribution (new feature,
documentation, bug fix, new tests, new examples, tutorial, etc) and assigning it to
yourself, this will help to inform others that you are working on the contribution
and if you want, it is also a vehicle to ask for help with it.
2. Fork the repo
3. Create your feature branch (`git checkout -b my-new-feature`)
4. Commit your changes (`git commit -s -m 'Informative commit message'`)
5. Push to the branch (`git push origin my-new-feature`)
6. Create new Pull Request

## License
This project is licensed under the Apache License 2.0. Each source file must include a copyright and license header for the Apache License 2.0. Using the SPDX format is the simplest approach.
e.g.

```c++
// Copyright contributors to the speakmin project
// SPDX-License-Identifier: Apache-2.0
```

## Signing off your contribution
This project uses [DCO][3]. Be sure to [sign off][4] your contributions using the `-s` flag or adding `Signed-off-By: Name<Email>` in the git commit message. e.g.

```bash
git commit -s -m 'Informative commit message'
```

  [1]: https://docs.github.com/en/github/using-git    "GitHubDocs"
  [2]: https://guides.github.com/introduction/git-handbook/    "gitHandbook"
  [3]: https://developercertificate.org/    "DCO"
  [4]: https://docs.github.com/en/github/authenticating-to-github/signing-commits    "gitSignoff"
