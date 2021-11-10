## Contributing to Bagua

Refer to the following guidelines to contribute new functionality or bug fixes to Bagua:

1. Create or follow an issue to discuss.
2. Write code and pass existing CI for correctness and format consistence.
3. Add unit tests for any new code you write.
4. Pass unit tests in both CPU and GPU environments.

Pull request title should follow [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/) style. Bagua current allows the following scopes:

- `core`
- `python`
- `net`

Any non-trivial pull requests will go through the refinement process to get the approval review from 

* the project administrator (https://github.com/NOBLES5E) for API and documentation check, and
* the corresponding code owners (as defined in [CODEOWNERS](https://github.com/BaguaSys/bagua/blob/master/.github/CODEOWNERS)) for implementation and tests check

before merging.

> Note: In general, breaking change should be avoided. If breaking changed is needed, it should be explicitly shown in the pull request description and go through the same code review process.
