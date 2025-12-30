import graphene
from graphene_django import DjangoObjectType
from django.contrib.auth.models import User
from .models import Post

class PostType(DjangoObjectType):
    class Meta:
        model = Post
        fields = ("id", "title", "body", "author")


class UserType(DjangoObjectType):
    class Meta:
        model = User
        fields = ("id", "username", "email", "posts")

    posts_count = graphene.Int()

    def resolve_posts_count(root, info):
        return root.posts.count()

class CreateUser(graphene.Mutation):
    class Arguments:
        username = graphene.String(required=True)
        email = graphene.String(required=True)
        password = graphene.String(required=True)

    user = graphene.Field(UserType)

    def mutate(root, info, username, email, password):
        user = User(username=username, email=email)
        user.set_password(password)  # Хешуємо пароль
        user.save()
        return CreateUser(user=user)

class UpdateUser(graphene.Mutation):
    class Arguments:
        id = graphene.ID(required=True)
        username = graphene.String()
        email = graphene.String()

    user = graphene.Field(UserType)
    message = graphene.String()

    def mutate(root, info, id, username=None, email=None):
        try:
            user = User.objects.get(pk=id)
            if username:
                user.username = username
            if email:
                user.email = email
            user.save()
            return UpdateUser(user=user, message="Success")
        except User.DoesNotExist:
            return UpdateUser(user=None, message="User not found")

class DeleteUser(graphene.Mutation):
    class Arguments:
        id = graphene.ID(required=True)

    success = graphene.Boolean()

    def mutate(root, info, id):
        try:
            user = User.objects.get(pk=id)
            user.delete()
            return DeleteUser(success=True)
        except User.DoesNotExist:
            return DeleteUser(success=False)

class Mutation(graphene.ObjectType):
    create_user = CreateUser.Field()
    update_user = UpdateUser.Field()
    delete_user = DeleteUser.Field()

class Query(graphene.ObjectType):
    all_users = graphene.List(UserType)
    user = graphene.Field(UserType, id=graphene.Int())

    def resolve_all_users(root, info):
        return User.objects.all()

    def resolve_user(root, info, id):
        return User.objects.get(pk=id)

schema = graphene.Schema(query=Query, mutation=Mutation)
